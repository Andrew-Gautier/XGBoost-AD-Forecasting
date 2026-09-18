# =============================================================================
# leadtime.py — Lead‑time analysis for early prediction models
# =============================================================================
# This module provides functions to:
#   - Run a trained classifier on truncated visit sequences (progressors & controls)
#   - Compute alarm probabilities/predictions for each truncation
#   - Evaluate detection sensitivity, specificity, lead times and false alarm rates
#   - Aggregate results across models, thresholds, and mask lengths
# =============================================================================

import os
import glob
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score

from feature_engineering import create_delta_features, preprocess_data
from preprocessing import (
    transform_imputer,
    apply_hv_composites,
    _parse_list_col,
)
from preprocessing import _LONG_COLS as PREPROC_LONG_COLS

# -----------------------------------------------------------------------------
# Constants & utilities
# -----------------------------------------------------------------------------

# Columns that are stored as lists (time‑varying features) in the input CSV
_LONG_COLS = [
    'NACCBMI', 'NACCMMSE', 'NACCGDS', 'CDRSUM', 'TOBAC30',
    'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
    'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL',
    'hearing', 'vision',
    'NACCLIVS', 'ALCOHOL', 'COMMUN', 'months_since_baseline'
]


def eval_if_str(x):
    """If x is a string containing a Python literal, evaluate it."""
    if isinstance(x, str):
        return eval(x)
    return x


def _parse_list(s):
    """Parse a stringified list (e.g. for 'months_since_baseline' or 'Progression')."""
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


def _pred_cols(df):
    """Return the prediction column names (like '0', '1', ...) sorted numerically."""
    cols = [c for c in df.columns if c not in ('ID', 'months_since_baseline', 'Progression')]
    return sorted(cols, key=lambda x: int(x))


def mean_ci(data, conf=0.95):
    """Compute mean, lower/upper confidence bounds, and std for a 1D array."""
    n = len(data)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    sem = std / np.sqrt(n)
    ci = stats.t.ppf((1 + conf) / 2, df=n - 1) * sem
    return mean, mean - ci, mean + ci, std


def get_conversion_month(progression, months):
    """
    Find the month of the first visit where progression label increases.
    progression : list of ints (e.g. [0,0,1,2])
    months      : list of corresponding month values
    Returns the month of the first visit with a higher value than the initial.
    """
    start = progression[0]
    for i, val in enumerate(progression):
        if val > start:
            return months[i]
    # Fallback (should not happen for progressors)
    return months[-1]


# -----------------------------------------------------------------------------
# Data transformation: create truncated feature sets
# -----------------------------------------------------------------------------

def transform(row, progression_type, mask_length=0):
    """
    Generate a DataFrame of engineered features for all truncations of one subject.

    Parameters
    ----------
    row : pandas Series
        A single row from the lead‑time CSV (includes list‑valued columns).
    progression_type : str
        'CN' or 'AD' – used by `preprocess_data`.
    mask_length : int
        Number of latest visits to mask (i.e., exclude from the end).

    Returns
    -------
    num_trunc : int
        Number of truncations generated.
    df : pandas DataFrame
        Transformed features (with one row per truncation).
    """
    num_trunc = row['n_visits'] - 1 - mask_length
    out = []
    # Start from 2 visits (minimum required)
    for length in range(2, num_trunc + 2):
        new_row = {}
        for col in row.index:
            if col in _LONG_COLS:
                if isinstance(row[col], list):
                    new_row[col] = row[col][:length]
                else:
                    new_row[col] = np.nan
            else:
                new_row[col] = row[col]
        out.append(new_row)
    df = pd.DataFrame(out)
    df = create_delta_features(df)
    df, _, _ = preprocess_data(df, progression_type)
    return num_trunc, df.drop(columns=['target'])


# -----------------------------------------------------------------------------
# Core lead‑time analysis: run a model on progressors and controls
# -----------------------------------------------------------------------------

def run_leadtime(file_path, dest_dir, model, progression_type, mask_length=0, imputer_meta=None):
    """
    Run a trained classifier on all truncated visit sequences of progressors and controls.

    Saves probability and prediction CSVs for progressors and controls.

    Parameters
    ----------
    file_path : str
        Path to the lead‑time CSV (one row per subject).
    dest_dir : str
        Directory where CSV files will be written.
    model : sklearn‑compatible classifier
        Must have `predict_proba` and `predict` methods.
    progression_type : str
        'CN' or 'AD'.
    mask_length : int
        Number of latest visits to mask for control subjects only.
    imputer_meta : dict, optional
        Imputer metadata from preprocessing.fit_imputer (fitted on the training set).
        When provided, the raw holdout is imputed with that imputer (no leakage) and
        hearing/vision composites are built before feature engineering. When None and
        the CSV still contains raw hearing/vision columns, only the composites are built.
    """
    os.makedirs(dest_dir, exist_ok=True)
    csv = pd.read_csv(file_path)

    # --- Optional leak-free preprocessing -------------------------------------------------
    # The leak-free lead-time CSV is stored in the RAW schema (raw hearing/vision columns,
    # sentinel-cleaned but NOT imputed). Parse the raw list columns, then either apply the
    # train-fitted imputer, or just build the hearing/vision composites (non-imputed path).
    raw_hv_present = any(c in csv.columns for c in ('HEARING', 'VISION'))
    if imputer_meta is not None or raw_hv_present:
        raw_list_cols = [c for c in PREPROC_LONG_COLS if c in csv.columns]
        for col in raw_list_cols:
            csv[col] = csv[col].apply(_parse_list_col)
        if 'months_since_baseline' in csv.columns:
            csv['months_since_baseline'] = csv['months_since_baseline'].apply(_parse_list_col)
        if imputer_meta is not None:
            csv = transform_imputer(imputer_meta, csv)
        else:
            csv = apply_hv_composites(csv)

    # Parse list columns (no-op for columns already parsed above).
    # Use _parse_list_col (not plain eval) so that 'nan' placeholders inside
    # stringified lists are converted to np.nan instead of raising NameError.
    for col in _LONG_COLS:
        if col in csv.columns:
            csv[col] = csv[col].apply(_parse_list_col)

    p_df = csv[csv['Prog_ID'] == 1]
    np_df = csv[csv['Prog_ID'] == 0]

    # ---- Progressors ----
    prob_csv = []
    pred_csv = []
    for _, row in p_df.iterrows():
        new_row_prob = {'ID': row['ID'],
                        'months_since_baseline': row['months_since_baseline'],
                        'Progression': row['Progression']}
        new_row_pred = new_row_prob.copy()

        n, df = transform(row, progression_type)
        x = df.values
        prob = model.predict_proba(x)[:, 1]
        pred = model.predict(x)
        for i in range(n):
            new_row_prob[i] = prob[i]
            new_row_pred[i] = pred[i]
        prob_csv.append(new_row_prob)
        pred_csv.append(new_row_pred)

    prob_path = os.path.join(dest_dir, 'lead_time_probabilities.csv')
    pred_path = os.path.join(dest_dir, 'lead_time_predictions.csv')
    pd.DataFrame(prob_csv).to_csv(prob_path, index=False)
    pd.DataFrame(pred_csv).to_csv(pred_path, index=False)

    # ---- Controls ----
    prob_csv = []
    pred_csv = []
    for _, row in np_df.iterrows():
        if row['n_visits'] < mask_length + 2:
            continue

        new_row_prob = {'ID': row['ID'],
                        'months_since_baseline': row['months_since_baseline'],
                        'Progression': row['Progression']}
        new_row_pred = new_row_prob.copy()

        n, df = transform(row, progression_type, mask_length)
        x = df.values
        prob = model.predict_proba(x)[:, 1]
        pred = model.predict(x)
        for i in range(n):
            new_row_prob[i] = prob[i]
            new_row_pred[i] = pred[i]
        prob_csv.append(new_row_prob)
        pred_csv.append(new_row_pred)

    prob_path = os.path.join(dest_dir, 'control_probabilities.csv')
    pred_path = os.path.join(dest_dir, 'control_predictions.csv')
    pd.DataFrame(prob_csv).to_csv(prob_path, index=False)
    pd.DataFrame(pred_csv).to_csv(pred_path, index=False)


# -----------------------------------------------------------------------------
# Analysis of prediction CSVs (for a single mask_length=0 run)
# -----------------------------------------------------------------------------

def load_leadtime_results(base_dir):
    """Load progressor and control prediction CSVs from a run_leadtime output directory."""
    prog_pred = pd.read_csv(os.path.join(base_dir, 'lead_time_predictions.csv'))
    ctrl_pred = pd.read_csv(os.path.join(base_dir, 'control_predictions.csv'))
    for df in (prog_pred, ctrl_pred):
        df['months_since_baseline'] = df['months_since_baseline'].apply(_parse_list)
        df['Progression'] = df['Progression'].apply(_parse_list)
    return prog_pred, ctrl_pred


# -----------------------------------------------------------------------------
# Model loading and probability caching (for multiple models)
# -----------------------------------------------------------------------------

def load_models(experiment_dir, progression_type):
    """
    Load all trained models for a given progression_type from an experiment folder.

    Files are assumed to be named like "*_model_<progression_type>.pkl".
    Returns a dict {model_key: model_object}.
    """
    pattern = f"*_model_{progression_type}.pkl"
    model_paths = glob.glob(os.path.join(experiment_dir, pattern))
    models = {}
    for pkl in model_paths:
        key = os.path.basename(pkl).replace(f"_model_{progression_type}.pkl", "")
        models[key] = joblib.load(pkl)
    return models


def get_probabilities(models_dict, leadtime_csv, prog_type, cache_dir="leadtime_cache"):
    """
    For each model, either load previously saved probability CSVs or generate them.

    Returns a dict {model_key: {'prog': prog_prob_df, 'ctrl': ctrl_prob_df}}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    results = {}
    for key, model in models_dict.items():
        model_cache = os.path.join(cache_dir, key)
        os.makedirs(model_cache, exist_ok=True)
        prog_prob_path = os.path.join(model_cache, 'lead_time_probabilities.csv')
        ctrl_prob_path = os.path.join(model_cache, 'control_probabilities.csv')

        if os.path.exists(prog_prob_path) and os.path.exists(ctrl_prob_path):
            prog_df = pd.read_csv(prog_prob_path)
            ctrl_df = pd.read_csv(ctrl_prob_path)
        else:
            run_leadtime(leadtime_csv, model_cache, model, prog_type, mask_length=0)
            prog_df = pd.read_csv(prog_prob_path)
            ctrl_df = pd.read_csv(ctrl_prob_path)

        # Parse list columns
        for df in (prog_df, ctrl_df):
            df['months_since_baseline'] = df['months_since_baseline'].apply(_parse_list)
            df['Progression'] = df['Progression'].apply(_parse_list)
        results[key] = {'prog': prog_df, 'ctrl': ctrl_df}
    return results


# -----------------------------------------------------------------------------
# Grid evaluation over thresholds and mask lengths
# -----------------------------------------------------------------------------

def analyze_run(prog_prob_df, ctrl_prob_df, threshold, mask_length):
    """
    Compute metrics for one model's probability DataFrames.
    mask_length is applied ONLY to controls (held‑out visits from the end).
    Progressors are evaluated on all truncations up to n_visits‑2.
    Returns a dictionary with counts and performance metrics,
    including separate categories: early, at_conversion, late, missed,
    and both early-only and net (including late) lead times.
    """
    pred_cols = sorted([c for c in prog_prob_df.columns if c.isdigit()], key=int)

    # ---- Progressors ----
    prog_scores = []   # max probability per progressor (subject-level risk score)
    prog_records = []
    for _, row in prog_prob_df.iterrows():
        months = row['months_since_baseline']
        progression = row['Progression']
        true_conv_month = get_conversion_month(progression, months)
        n_visits = len(months)

        # Mask NOT applied to progressors – use all truncations
        max_trunc = n_visits - 2
        probs = []
        for c in pred_cols:
            if int(c) > max_trunc:
                continue
            if pd.notna(row[c]):
                probs.append(row[c])
        prog_scores.append(max(probs) if probs else np.nan)

        # Find first alarm (probability >= threshold)
        alarm_idx = None
        for i, p in enumerate(probs):
            if p >= threshold:
                alarm_idx = i
                break

        if alarm_idx is None:
            status = 'missed'
            alarm_month = np.nan
            lead_time = np.nan
        else:
            alarm_month = months[alarm_idx + 1]   # +1 because truncation i uses first i+2 visits
            lead_time = true_conv_month - alarm_month
            if lead_time > 0:
                status = 'detected_early'
            elif lead_time == 0:
                status = 'detected_at_conversion'
            else:
                status = 'detected_late'

        prog_records.append({
            'true_conversion_month': true_conv_month,
            'alarm_month': alarm_month,
            'lead_time': lead_time,
            'status': status,
        })
    prog_df = pd.DataFrame(prog_records)

    # ---- Controls ----
    ctrl_records = []
    ctrl_scores = []   # max probability per included control (subject-level risk score)
    for _, row in ctrl_prob_df.iterrows():
        months = row['months_since_baseline']
        n_visits = len(months)
        # Apply mask to controls
        max_trunc = n_visits - mask_length - 2
        if max_trunc < 0:
            continue
        probs = []
        for i in range(max_trunc + 1):
            c = str(i)
            if c in row and pd.notna(row[c]):
                probs.append(row[c])
        if not probs:
            continue
        ctrl_scores.append(max(probs))
        alarms = [p >= threshold for p in probs]
        n_alarms = sum(alarms)
        ctrl_records.append({
            'n_visits_used': len(probs),
            'n_alarms': n_alarms,
            'any_alarm': n_alarms > 0,
        })
    ctrl_df = pd.DataFrame(ctrl_records)

    # ---- Aggregate ----
    n_prog = len(prog_df)
    n_early   = (prog_df['status'] == 'detected_early').sum()
    n_at_conv = (prog_df['status'] == 'detected_at_conversion').sum()
    n_late    = (prog_df['status'] == 'detected_late').sum()
    n_missed  = (prog_df['status'] == 'missed').sum()
    n_detected = n_early + n_at_conv + n_late
    sensitivity = n_detected / n_prog if n_prog else np.nan

    # Early-only lead times (positive)
    early_lead_times = prog_df.loc[prog_df['status'] == 'detected_early', 'lead_time']
    mean_lead = early_lead_times.mean() if len(early_lead_times) else np.nan
    median_lead = early_lead_times.median() if len(early_lead_times) else np.nan

    # Net lead times (all detections, including negative late ones)
    all_detected_lead = prog_df.loc[prog_df['status'] != 'missed', 'lead_time']  # includes early, at_conv, late
    mean_net_lead = all_detected_lead.mean() if len(all_detected_lead) else np.nan
    median_net_lead = all_detected_lead.median() if len(all_detected_lead) else np.nan

    n_ctrl = len(ctrl_df)
    n_false_alarm = ctrl_df['any_alarm'].sum() if n_ctrl else 0
    specificity = 1 - n_false_alarm / n_ctrl if n_ctrl else np.nan
    false_alarm_rate = n_false_alarm / n_ctrl if n_ctrl else np.nan

    # ---- Subject-level discrimination (progressor=1 vs control=0) ----
    # Score = max probability across each subject's allowed truncations (the same
    # "ever fires" signal the threshold-based alarm rule uses). ROC-AUC / PR-AUC are
    # threshold-independent but depend on the control mask.
    auc_labels = np.array([1] * len(prog_scores) + [0] * len(ctrl_scores), dtype=int)
    auc_scores = np.array(prog_scores + ctrl_scores, dtype=float)
    valid = ~np.isnan(auc_scores)
    if valid.sum() >= 2 and len(np.unique(auc_labels[valid])) == 2:
        roc_auc = roc_auc_score(auc_labels[valid], auc_scores[valid])
        pr_auc = average_precision_score(auc_labels[valid], auc_scores[valid])
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        'n_prog': n_prog,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'n_detected': n_detected,
        'n_early': n_early,
        'n_at_conv': n_at_conv,
        'n_late': n_late,
        'n_missed': n_missed,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mean_lead_time': mean_lead,
        'median_lead_time': median_lead,
        'mean_net_lead_time': mean_net_lead,
        'median_net_lead_time': median_net_lead,
        'false_alarm_rate': false_alarm_rate,
        'n_ctrl': n_ctrl,
        'n_false_alarm': n_false_alarm,
    }

def run_grid(prob_dict, thresholds, mask_lengths):
    """
    Evaluate a set of models across thresholds and mask lengths.

    Parameters
    ----------
    prob_dict : dict
        As returned by get_probabilities: {model_key: {'prog': df, 'ctrl': df}}
    thresholds : list of float
    mask_lengths : list of int

    Returns
    -------
    DataFrame with columns: model_key, threshold, mask_length, metric...
    """
    rows = []
    for key, dfs in prob_dict.items():
        for thr in thresholds:
            for mask in mask_lengths:
                res = analyze_run(dfs['prog'], dfs['ctrl'], thr, mask)
                rows.append({
                    'model_key': key,
                    'threshold': thr,
                    'mask_length': mask,
                    **res
                })
    return pd.DataFrame(rows)


def aggregate_grid(df_grid):
    """Group grid results by threshold and mask_length, compute mean, std, CI for all metrics."""
    group_cols = ['threshold', 'mask_length']
    metrics = ['sensitivity', 'specificity', 'mean_lead_time', 'median_lead_time',
               'mean_net_lead_time', 'median_net_lead_time',
               'false_alarm_rate',
               'n_early', 'n_at_conv', 'n_late', 'n_missed', 'n_false_alarm']
    agg_list = []
    for (thr, mask), group in df_grid.groupby(group_cols):
        row = {'threshold': thr, 'mask_length': mask}
        for m in metrics:
            vals = group[m].dropna().values
            if len(vals) > 1:
                mean, lo, hi, std = mean_ci(vals)
                row[f'{m}_mean'] = mean
                row[f'{m}_std'] = std
                row[f'{m}_ci_lower'] = lo
                row[f'{m}_ci_upper'] = hi
            elif len(vals) == 1:
                row[f'{m}_mean'] = vals[0]
                row[f'{m}_std'] = np.nan
                row[f'{m}_ci_lower'] = np.nan
                row[f'{m}_ci_upper'] = np.nan
            else:
                row[f'{m}_mean'] = np.nan
                row[f'{m}_std'] = np.nan
                row[f'{m}_ci_lower'] = np.nan
                row[f'{m}_ci_upper'] = np.nan
        agg_list.append(row)
    return pd.DataFrame(agg_list)

# -----------------------------------------------------------------------------
# Example / test block (commented out by default)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage (uncomment and adapt paths as needed)
    #
    # import joblib
    # model = joblib.load('path/to/model.pkl')
    # run_leadtime(
    #     'path/to/lead_time.csv',
    #     'output_dir',
    #     model,
    #     'AD',
    #     mask_length=0
    # )
    pass