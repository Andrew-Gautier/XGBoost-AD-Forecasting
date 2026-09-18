import numpy as np
import pandas as pd
from scipy.stats import linregress
from preprocessing import create_target


# Initially 40 columns from preprocessing

LONGITUDINAL_COLUMNS = ['NACCBMI', 'NACCMMSE', 'NACCGDS', 'CDRSUM', 'TOBAC30',
    'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
    'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL',
    'NACCLIVS', 'COMMUN', 'hearing', 'vision', 'ALCOHOL']  # 20 features


BINARY_COLUMNS = ['NACCFAM',
    'CVHATT', 'CVAFIB', 'DIABETES', 'HYPERCHO', 'HYPERTEN',
    'B12DEF', 'DEPD', 'ANX', 'NACCTBI', 'HISPANIC'] 

NUMERIC_COLUMNS = ['EDUC', 'SMOKYRS', 'age']  

CATEGORICAL_COLUMNS = ['SEX', 'RACE', 'NACCNE4S']  

# 37 features in the end: 17 static + 20 longitudinal
def preprocess_data(df, progression_type):
    # Create target variable if not already present
    if 'target' not in df.columns:
        df['target'] = df['Progression'].apply(create_target, progression_type=progression_type)
    
    # Get all available features (after create_delta_features transformation)
    all_features = [col for col in df.columns if col != 'target']
    
    # Select features we want to keep
    # Static features (non-time-series)
    static_features = [
        'SEX', 'EDUC', 'NACCFAM', 'CVHATT', 
        'CVAFIB', 'DIABETES', 'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 
        'ANX', 'NACCTBI', 'SMOKYRS', 'RACE', 'age', 'HISPANIC', 'NACCNE4S'
    ]
    
    # Time-series features — visit-agnostic summary statistics
    _TS_SUFFIXES = ('_slope', '_mean', '_max', '_min', '_std', '_range',
                    '_first', '_last', '_last_minus_first', '_acceleration',
                    '_n_visits', '_pct_change', '_std_slope',
                    '_lag1', '_lag2', '_lag3',
                    '_interval_mean', '_interval_std')

    # Hearing × Vision interaction features (named explicitly — not reliably caught by suffixes)
    _INTERACTION_FEATURES = ['hearing_vision_product', 'hearing_vision_sum', 'hearing_vision_mean']
    interaction_features = [f for f in _INTERACTION_FEATURES if f in df.columns]

    # Exclude interaction features from the suffix scan to avoid duplicates
    # (e.g. 'hearing_vision_mean' matches '_mean' but is already in interaction_features)
    time_series_features = [col for col in all_features
                            if any(s in col for s in _TS_SUFFIXES)
                            and col not in _INTERACTION_FEATURES]

    # Only keep features that actually exist in the dataframe
    static_features = [f for f in static_features if f in df.columns]

    # Combine all features
    features = static_features + time_series_features + interaction_features
    
    # Handle missing values
    df = df[features + ['target']].copy()
    df = df.dropna(subset=features, how='all')
    
    # Convert true categorical columns to pandas category dtype.
    # XGBoost (enable_categorical=True) will treat them as unordered categoricals.
    # Binary and numeric columns are left as-is.
    categorical_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Return processed DataFrame without imputation/scaling to avoid leakage
    # Keep return signature compatibility (scaler, imputer as None)
    return df, None, None



def _parse_array_columns(df):
    """Convert array-string columns (e.g. '[1.0, 2.5, nan]') to lists of floats in-place."""
    for col in df.columns:
        # Compatible with both legacy object dtype and pandas 2.0+ StringDtype
        try:
            has_array = df[col].str.startswith('[', na=False).any()
        except AttributeError:
            continue
        if not has_array:
            continue
        df[col] = df[col].apply(
            lambda x: [float(v.strip()) if v.strip() != 'nan' else np.nan
                       for v in x.replace('[', '').replace(']', '').split(',')]
            if isinstance(x, str) and x.startswith('[') else np.nan
        )


def _is_numeric_list_col(series):
    """True if the first element is a list of numeric values."""
    first = series.iloc[0]
    return (isinstance(first, list)
            and all(isinstance(v, (int, float)) for v in first if not pd.isna(v)))


def _has_valid(x):
    return isinstance(x, list) and np.any(~np.isnan(x))


def _first_valid(x):
    if not isinstance(x, list):
        return np.nan
    return next((v for v in x if not np.isnan(v)), np.nan)


def _last_valid(x):
    if not isinstance(x, list):
        return np.nan
    return next((v for v in reversed(x) if not np.isnan(v)), np.nan)


def _calc_slope(x, t=None):
    """Linear slope of x.  When t is provided (real elapsed months), uses t as the
    x-axis so the slope is in units-per-month rather than units-per-visit-index."""
    if not (isinstance(x, list) and len(x) > 1):
        return np.nan
    yv = np.array(x, dtype=float)
    valid = ~np.isnan(yv)
    if t is not None:
        tv = np.array(t, dtype=float)
        valid = valid & ~np.isnan(tv)
        if valid.sum() > 1:
            return linregress(tv[valid], yv[valid]).slope
    else:
        if valid.sum() > 1:
            xv = np.arange(len(x))
            return linregress(xv[valid], yv[valid]).slope
    return np.nan


def _calc_acceleration(x, t=None):
    """Slope of the rate-of-change (2nd derivative). Requires >= 4 non-NaN values.

    When t is provided (real elapsed months), pairwise slopes are computed as
    (y[i+1]-y[i])/(t[i+1]-t[i]) and regressed over the midpoint times, giving
    a time-normalized acceleration.  Falls back to visit-index-based computation
    when t is None.
    """
    if not isinstance(x, list) or len(x) < 4:
        return np.nan
    arr = np.array(x, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 4:
        return np.nan
    vals = arr[valid]
    if t is not None:
        tv = np.array(t, dtype=float)
        tv_valid = tv[valid]
        dt = np.diff(tv_valid)
        dy = np.diff(vals)
        nonzero = dt != 0
        if nonzero.sum() < 3:
            return np.nan
        pairwise_slopes = dy[nonzero] / dt[nonzero]
        t_mid = ((tv_valid[:-1] + tv_valid[1:]) / 2)[nonzero]
        if len(pairwise_slopes) < 2:
            return np.nan
        return linregress(t_mid, pairwise_slopes).slope
    else:
        deltas = np.diff(vals)
        if len(deltas) < 3:
            return np.nan
        idx = np.arange(len(deltas))
        return linregress(idx, deltas).slope


def _calc_pct_change(x):
    """Maximum absolute percentage change between consecutive valid visits.

    Returns NaN if fewer than 2 valid values or all denominators are zero.
    """
    if not isinstance(x, list):
        return np.nan
    arr = np.array(x, dtype=float)
    valid_vals = arr[~np.isnan(arr)]
    if len(valid_vals) < 2:
        return np.nan
    diffs = np.abs(np.diff(valid_vals))
    denoms = np.abs(valid_vals[:-1])
    nonzero = denoms != 0
    if not nonzero.any():
        return np.nan
    return float(np.max(diffs[nonzero] / denoms[nonzero]))


def _calc_std_slope(x, t):
    """Standard deviation of pairwise per-month slopes between consecutive valid visits.

    Returns 0.0 if exactly 1 slope exists (2 valid visits — no variability).
    Returns NaN if fewer than 2 valid visits or all time intervals are zero.
    """
    if not (isinstance(x, list) and isinstance(t, list)):
        return np.nan
    arr_x = np.array(x, dtype=float)
    arr_t = np.array(t, dtype=float)
    valid = ~np.isnan(arr_x) & ~np.isnan(arr_t)
    if valid.sum() < 2:
        return np.nan
    vx = arr_x[valid]
    vt = arr_t[valid]
    dt = np.diff(vt)
    dy = np.diff(vx)
    nonzero = dt != 0
    if nonzero.sum() < 1:
        return np.nan
    slopes = dy[nonzero] / dt[nonzero]
    if len(slopes) == 1:
        return 0.0
    return float(np.std(slopes, ddof=1))


def _calc_interval_mean(t):
    """Mean of consecutive time differences (visit intervals in months)."""
    if not isinstance(t, list) or len(t) < 2:
        return np.nan
    arr = np.array(t, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return np.nan
    return float(np.mean(np.diff(valid)))


def _calc_interval_std(t):
    """Std of consecutive time differences (irregularity of visit spacing)."""
    if not isinstance(t, list) or len(t) < 3:
        return np.nan
    arr = np.array(t, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 3:
        return np.nan
    diffs = np.diff(valid)
    if len(diffs) < 2:
        return np.nan
    return float(np.std(diffs, ddof=1))


def _last_nth_valid(x, n):
    """Return the nth-from-last valid (non-NaN) value in a list.

    n=1 → second-to-last (lag1), n=2 → third-to-last (lag2), etc.
    Returns NaN if fewer than n+1 valid values exist.
    """
    if not isinstance(x, list):
        return np.nan
    valid_vals = [v for v in x if not np.isnan(v)]
    idx = -(n + 1)
    if len(valid_vals) >= abs(idx):
        return valid_vals[idx]
    return np.nan


def _engineer_visit_agnostic(df, new_columns, months_series=None):
    """Populate *new_columns* dict with visit-agnostic features for all numeric-list columns in *df*.

    months_series : pd.Series, optional
        Each element is a list of elapsed months (one per visit) aligned to *df*.
        When provided, slopes and accelerations are time-normalized (units per month)
        rather than units per visit index.  Also enables _std_slope and _pct_change.
        The 'months_since_baseline' column is handled specially (interval stats only,
        no self-referential slope).
    """
    for col in df.columns:
        if not _is_numeric_list_col(df[col]):
            continue

        # ── Standard summary stats (same for all longitudinal columns) ────────
        new_columns[f"{col}_mean"] = df[col].apply(
            lambda x: np.nanmean(x) if _has_valid(x) else np.nan)
        new_columns[f"{col}_max"] = df[col].apply(
            lambda x: np.nanmax(x) if _has_valid(x) else np.nan)
        new_columns[f"{col}_min"] = df[col].apply(
            lambda x: np.nanmin(x) if _has_valid(x) else np.nan)
        new_columns[f"{col}_std"] = df[col].apply(
            lambda x: (np.nanstd(x, ddof=1)
                       if isinstance(x, list) and np.sum(~np.isnan(x)) > 1
                       else np.nan))
        new_columns[f"{col}_range"] = df[col].apply(
            lambda x: (np.nanmax(x) - np.nanmin(x)) if _has_valid(x) else np.nan)
        new_columns[f"{col}_first"] = df[col].apply(_first_valid)
        new_columns[f"{col}_last"] = df[col].apply(_last_valid)
        new_columns[f"{col}_last_minus_first"] = df[col].apply(
            lambda x: (_last_valid(x) - _first_valid(x))
            if _has_valid(x) else np.nan)
        new_columns[f"{col}_n_visits"] = df[col].apply(
            lambda x: int(np.sum(~np.isnan(x))) if isinstance(x, list) else np.nan)

        # ── months_since_baseline: interval stats only, no rate features ──────
        if col == 'months_since_baseline':
            new_columns['months_since_baseline_interval_mean'] = df[col].apply(_calc_interval_mean)
            new_columns['months_since_baseline_interval_std']  = df[col].apply(_calc_interval_std)
            continue

        # ── Rate / variability features (time-normalized when possible) ───────
        if months_series is not None:
            new_columns[f"{col}_slope"] = pd.Series(
                [_calc_slope(x, t) for x, t in zip(df[col], months_series)],
                index=df.index,
            )
            new_columns[f"{col}_acceleration"] = pd.Series(
                [_calc_acceleration(x, t) for x, t in zip(df[col], months_series)],
                index=df.index,
            )
            new_columns[f"{col}_std_slope"] = pd.Series(
                [_calc_std_slope(x, t) for x, t in zip(df[col], months_series)],
                index=df.index,
            )
        else:
            new_columns[f"{col}_slope"]        = df[col].apply(_calc_slope)
            new_columns[f"{col}_acceleration"] = df[col].apply(_calc_acceleration)

        new_columns[f"{col}_pct_change"] = df[col].apply(_calc_pct_change)


def create_delta_features(df):
    """Visit-agnostic feature engineering.

    For each longitudinal (list-valued) column, creates fixed-width summary
    statistics that are independent of the number of visits:

        _mean, _max, _min, _std, _range, _slope (time-normalized), _first, _last,
        _last_minus_first, _acceleration (>=4 visits), _n_visits,
        _pct_change, _std_slope, _lag1, _lag2, _lag3 (selected columns),
        months_since_baseline_interval_mean, months_since_baseline_interval_std,
        hearing_vision_product, hearing_vision_sum, hearing_vision_mean

    Static / scalar columns are passed through unchanged.
    """
    # Columns for which lag features are computed
    _LAG_COLUMNS = ('NACCMMSE', 'CDRSUM', 'NACCGDS', 'hearing', 'vision', 'BILLS', 'SHOPPING')

    df = df.copy()
    new_columns = {}

    _parse_array_columns(df)

    # ── ALCOHOL severity recode ───────────────────────────────────────────────
    # Raw NACC coding:  0=Absent, 1=Recent/Active, 2=Remote/Inactive
    # Problem: numeric order implies Remote/Inactive (2) > Recent/Active (1),
    # which inverts clinical severity.
    # Recode so that higher numbers always mean greater clinical concern:
    #   0 (Absent)          → 0
    #   2 (Remote/Inactive) → 1
    #   1 (Recent/Active)   → 2   ← most severe
    # A positive slope means worsening abuse history.
    _ALCOHOL_RECODE = {0.0: 0.0, 1.0: 2.0, 2.0: 1.0}
    if 'ALCOHOL' in df.columns and _is_numeric_list_col(df['ALCOHOL']):
        df['ALCOHOL'] = df['ALCOHOL'].apply(
            lambda lst: [_ALCOHOL_RECODE.get(v, v) if not (isinstance(v, float) and np.isnan(v)) else np.nan
                         for v in lst]
            if isinstance(lst, list) else lst
        )

    # Extract months_since_baseline as a companion series for time-normalized slopes.
    # It is itself a list column and will be processed (interval stats) inside
    # _engineer_visit_agnostic, but excluded from slope/acceleration computation.
    months_series = None
    if 'months_since_baseline' in df.columns and _is_numeric_list_col(df['months_since_baseline']):
        months_series = df['months_since_baseline']

    _engineer_visit_agnostic(df, new_columns, months_series=months_series)

    # ── Phase 3: Lagged features ──────────────────────────────────────────────
    for col in _LAG_COLUMNS:
        if col not in df.columns or not _is_numeric_list_col(df[col]):
            continue
        for n in (1, 2, 3):
            new_columns[f"{col}_lag{n}"] = df[col].apply(
                lambda x, _n=n: _last_nth_valid(x, _n)
            )

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # ── Phase 4: Hearing × Vision interaction features ────────────────────────
    if 'hearing_last' in df.columns and 'vision_last' in df.columns:
        h = df['hearing_last']
        v = df['vision_last']
        df['hearing_vision_product'] = h * v
        df['hearing_vision_sum']     = h + v
        df['hearing_vision_mean']    = (h + v) / 2

    # Drop original array columns (keep only engineered features)
    array_cols = [col for col in df.columns if isinstance(df[col].iloc[0], list)] if len(df) > 0 else []
    df = df.drop(columns=array_cols)

    return df