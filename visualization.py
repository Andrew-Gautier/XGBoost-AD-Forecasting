import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def get_model_feature_names(model, X):
    """Recover the ordered feature names a model was trained on.

    Prefers the stored ``feature_names_in_`` attribute (set explicitly during
    training), falls back to the XGBoost booster's feature names, and finally
    to generic ``f0, f1, ...`` labels so plotting never fails.
    """
    X = np.asarray(X)
    n_features = X.shape[1]
    try:
        fn = list(model.feature_names_in_)
        if len(fn) == n_features:
            return fn
    except Exception:
        pass
    try:
        fn = model.get_booster().feature_names
        if fn and len(fn) == n_features:
            return list(fn)
    except Exception:
        pass
    return [f"f{i}" for i in range(n_features)]


def recompute_run_feature_names(csv_path, progression_type, seed):
    """Recompute the ordered feature names for a non-imputed training run.

    Replays the exact preprocessing used by ``model.train_best_model`` so the
    returned names align 1:1 with the saved ``{key}_X_train.npy`` array columns
    (the pickled XGBoost model does not persist ``feature_names_in_`` because it
    was fit on a numpy array).
    """
    from sklearn.model_selection import train_test_split
    from preprocessing import create_target
    from feature_engineering import create_delta_features, preprocess_data

    dataset = pd.read_csv(csv_path)
    dataset['target'] = dataset['Progression'].apply(
        create_target, progression_type=progression_type
    )
    train_idx, _ = train_test_split(
        dataset.index, test_size=0.2, random_state=seed, stratify=dataset['target']
    )
    train = dataset.loc[train_idx].copy()
    y = train['target'].values
    train = train.drop(columns=['target'])
    train_feat = create_delta_features(train)
    train_feat['target'] = y
    processed, _, _ = preprocess_data(train_feat, progression_type)
    return [c for c in processed.columns if c != 'target']


def plot_feature_importance(importances, feature_names, top_n=50, title=None, save_path=None,
                            title_fontsize=14, label_fontsize=12, tick_fontsize=10, value_fontsize=8):
    """Publication-quality horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importances : array-like or fitted model
        Either a 1-D array of importance values or a fitted model with
        a `feature_importances_` attribute.
    feature_names : list[str]
        Feature names corresponding to the importance values.
    top_n : int
        Number of top features to display.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    title_fontsize : int
        Font size for the chart title.
    label_fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels (feature names).
    value_fontsize : int
        Font size for the value annotations on each bar.
    """
    if hasattr(importances, 'feature_importances_'):
        importances = importances.feature_importances_
    fi = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi = fi.sort_values('Importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(fi)))
    ax.barh(fi['Feature'], fi['Importance'], color=colors, edgecolor='white', linewidth=0.5)
    for i, (val, name) in enumerate(zip(fi['Importance'], fi['Feature'])):
        ax.text(val + fi['Importance'].max() * 0.01, i, f'{val:.4f}', va='center', fontsize=value_fontsize)
    ax.set_xlabel('Importance (gain)', fontsize=label_fontsize)
    ax.set_title(title or f'Top {top_n} Feature Importances', fontsize=title_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance_axis(ax, importances, feature_names, top_n=50,
                                 cmap='viridis', value_fontsize=7,
                                 label_fontsize=9, tick_fontsize=8):
    """Publication-quality horizontal bar chart of top-N importances on an axis.

    Draws into the provided ``ax`` (no figure creation) so CN-MCI and MCI-AD
    panels can be composed side by side. The standalone ``plot_feature_importance``
    remains unchanged for backward compatibility.
    """
    importances = np.asarray(importances, dtype=float)
    fi = pd.DataFrame({'Feature': list(feature_names), 'Importance': importances})
    fi = fi.sort_values('Importance', ascending=True).tail(top_n)

    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.85, len(fi)))
    ax.barh(fi['Feature'], fi['Importance'], color=colors,
            edgecolor='white', linewidth=0.3)
    max_imp = fi['Importance'].max()
    for i, (val, name) in enumerate(zip(fi['Importance'], fi['Feature'])):
        ax.text(val + max_imp * 0.012, i, f'{val:.4f}', va='center', ha='left',
                fontsize=value_fontsize)
    ax.set_xlabel('Importance (gain)', fontsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    return fi


def plot_aggregate_feature_importance_axis(ax, importances_matrix, feature_names,
                                           top_n=50, ci=95, cmap='viridis',
                                           value_fontsize=7, label_fontsize=9,
                                           tick_fontsize=8, error_bars=False):
    """Aggregate feature importances across multiple models on a provided axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    importances_matrix : array-like, shape (n_models, n_features)
        Feature importances for each model (e.g. XGBoost gain, already
        normalized per model). Rows must be aligned 1:1 with ``feature_names``.
    feature_names : list[str]
        Feature names, one per column of ``importances_matrix``.
    top_n : int
        Number of top features to display (ranked by mean importance).
    ci : float, default 95
        Confidence level for the ± interval / error bars (t-distribution across models).
    error_bars : bool, default False
        If True, draw the ``ci``% interval as error-bar whiskers and annotate the
        mean only. If False, annotate ``mean ± interval`` as text after each bar.

    Returns
    -------
    fi : pd.DataFrame
        The top-``top_n`` features with columns ``Feature``, ``Mean``, ``Err``
        (``Err`` = half-width of the ``ci``% confidence interval).
    """
    M = np.asarray(importances_matrix, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    n_models, n_features = M.shape
    assert n_features == len(feature_names), (n_features, len(feature_names))

    means = M.mean(axis=0)
    if n_models > 1:
        sem = M.std(axis=0, ddof=1) / np.sqrt(n_models)
        tcrit = stats.t.ppf((1 + ci / 100) / 2, df=n_models - 1)
        err = tcrit * sem
    else:
        err = np.zeros(n_features)

    fi = pd.DataFrame({'Feature': list(feature_names), 'Mean': means, 'Err': err})
    fi = fi.sort_values('Mean', ascending=True).tail(top_n)

    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.85, len(fi)))
    y = np.arange(len(fi))
    if error_bars:
        # Gain importance is non-negative, so clamp the lower whisker at 0
        # (a symmetric t-interval can dip below zero for high-variance features).
        mean_arr = fi['Mean'].to_numpy()
        err_arr = fi['Err'].to_numpy()
        xerr = np.vstack([np.minimum(err_arr, mean_arr), err_arr])
        ax.barh(y, mean_arr, xerr=xerr, color=colors,
                edgecolor='white', linewidth=0.3, capsize=2,
                error_kw=dict(elinewidth=0.6, capthick=0.6))
    else:
        ax.barh(y, fi['Mean'], color=colors,
                edgecolor='white', linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(fi['Feature'], fontsize=tick_fontsize)
    max_imp = fi['Mean'].max()
    if error_bars:
        for i, (val, e) in enumerate(zip(fi['Mean'], fi['Err'])):
            ax.text(val + e + max_imp * 0.012, i, f'{val:.4f}',
                    va='center', ha='left', fontsize=value_fontsize)
    else:
        for i, (val, e) in enumerate(zip(fi['Mean'], fi['Err'])):
            ax.text(val + max_imp * 0.012, i, f'{val:.4f} ± {e:.4f}',
                    va='center', ha='left', fontsize=value_fontsize)
    ax.set_xlabel('Importance (gain)', fontsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    return fi


def plot_confusion_mat(y_true, y_pred, class_labels=None, title=None, save_path=None):
    """Publication-quality annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    class_labels : list[str], optional
        Display names for classes (default: ['0', '1']).
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_true))]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
                yticklabels=class_labels, linewidths=0.5, linecolor='gray',
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title or 'Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc(y_true, y_proba, title=None, save_path=None):
    """Publication-quality ROC curve with AUC annotation and diagonal reference.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'ROC curve (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2E86AB')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title or 'Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fpr, tpr, auc_val


def plot_pr_curve(y_true, y_proba, title=None, save_path=None):
    """Publication-quality Precision-Recall curve with Average Precision annotation.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#E84855', lw=2.5, label=f'PR curve (AP = {ap:.3f})')
    ax.axhline(baseline, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Random classifier (AP = {baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.15, color='#E84855')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title or 'Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return precision, recall, ap


def plot_shap_summary(model, X_train, feature_names, max_display=50, title=None,
                      save_path=None, xlim=None, figsize=(10, 6)):
    """
    Parameters
    ----------
    ...
    xlim : tuple or None, default None
        If provided, set the x-axis limits, e.g., (-1.0, 1.0).
    figsize : tuple, default (10, 6)
        Figure size (width, height) in inches.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    ax = plt.gca()

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    fig.set_size_inches(figsize[0], figsize[1])

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return shap_values, explainer


def _swarm_offsets(x, spread=0.38, n_bins=100, seed=0):
    """Density-adaptive vertical jitter (in feature-row units) for one beeswarm row.

    Mirrors the packing algorithm used by ``shap.plots.beeswarm``: points are
    bucketed into ``n_bins`` value-bins and stacked with an alternating
    +/- layer order (0, +1, -1, +2, -2, ...), then the *entire row* is rescaled
    by its own densest bin (``spread / (max_stack + 1)``) instead of using a
    fixed per-layer gap with hard clipping. This lets sparse bins spread out
    and dense bins compress smoothly, avoiding the flat "shelf" banding that a
    fixed-gap + clip approach produces. A tiny random tie-break (seeded, so
    output stays reproducible) is added to the bin-sort key so same-layer dots
    across adjacent bins don't align into visible grid stripes.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        bin_idx = np.zeros(n, dtype=int)
    else:
        edges = np.linspace(xmin, xmax, n_bins + 1)
        bin_idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, n_bins - 1)

    rng = np.random.default_rng(seed)
    sort_key = bin_idx + rng.normal(0.0, 1e-6, size=n)
    order = np.argsort(sort_key)

    raw = np.zeros(n)
    layer = 0
    last_bin = -1
    for idx in order:
        b = bin_idx[idx]
        if b != last_bin:
            layer = 0
        raw[idx] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = b

    max_stack = raw.max() if n else 0.0
    scale = 0.9 * spread / (max_stack + 1)
    return raw * scale


def plot_shap_beeswarm(ax, shap_values, X, feature_names, top_n=50,
                       cmap='coolwarm', dot_size=5, alpha=0.85,
                       xlabel="SHAP value (impact on model output)",
                       label_fontsize=9, tick_fontsize=8):
    """Publication-quality horizontal SHAP beeswarm on a provided axis.

    Ranks features by mean |SHAP| and shows the top ``top_n``. Each dot is a
    sample, coloured by that sample's feature value (blue = low, red = high).
    The standalone ``plot_shap_summary`` remains unchanged for compatibility.
    """
    shap_values = np.asarray(shap_values, dtype=float)
    X = np.asarray(X, dtype=float)
    feature_names = list(feature_names)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]

    cmap_obj = plt.get_cmap(cmap)
    for row, feat_idx in enumerate(order):
        vals = X[:, feat_idx]
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmax - vmin < 1e-12:
            norm = np.full(vals.shape, 0.5)
        else:
            norm = (vals - vmin) / (vmax - vmin)
        sv = shap_values[:, feat_idx]
        y = row + _swarm_offsets(sv)
        ax.scatter(sv, y, c=norm, cmap=cmap_obj, s=dot_size, alpha=alpha,
                   linewidths=0, vmin=0.0, vmax=1.0)

    ax.axvline(0.0, color='0.4', lw=0.8, zorder=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=tick_fontsize)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    return order, mean_abs


def plot_aggregate_roc(y_true_list, y_proba_list, label, color='#2E86AB',
                       n_points=100, ci=95, save_path=None, ax=None):
    """
    Plot aggregate ROC curve from multiple (y_true, y_proba) pairs.

    Parameters
    ----------
    y_true_list : list of array-like
        True labels for each run.
    y_proba_list : list of array-like
        Predicted probabilities for each run.
    label : str
        Legend label for the curve.
    color : str
        Main color for the line and fill.
    n_points : int
        Number of FPR points for interpolation.
    ci : float, default 95
        Confidence interval percentage (e.g., 95 for 95% band).
    save_path : str, optional
        If given, save the figure.
    ax : matplotlib.axes, optional
        If provided, plot on this axis; otherwise create new figure.
    """
    # Compute AUCs for each run (optional, for annotation)
    aucs = [roc_auc_score(yt, yp) for yt, yp in zip(y_true_list, y_proba_list)]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs, ddof=1)

    # Common FPR grid
    fpr_grid = np.linspace(0, 1, n_points)
    tprs = []

    for yt, yp in zip(y_true_list, y_proba_list):
        fpr, tpr, _ = roc_curve(yt, yp)
        # Interpolate TPR to the common FPR grid
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tprs.append(tpr_interp)

    tprs = np.array(tprs)  # shape: (n_runs, n_points)

    mean_tpr = tprs.mean(axis=0)
    lower = np.percentile(tprs, (100 - ci) / 2, axis=0)
    upper = np.percentile(tprs, 100 - (100 - ci) / 2, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr_grid, mean_tpr, color=color, lw=2.5,
            label=f'{label} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(fpr_grid, lower, upper, color=color, alpha=0.2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return fpr_grid, mean_tpr, lower, upper, mean_auc, std_auc


def plot_aggregate_pr(y_true_list, y_proba_list, label, color='#E84855',
                      n_points=100, ci=95, save_path=None, ax=None,
                      plot_baseline=True):
    """Plot aggregate precision-recall curve from multiple (y_true, y_proba) pairs.

    plot_baseline : bool, default True
        If False, skip the dashed horizontal random-classifier line (and its
        legend entry), e.g. when the chance-level AP is reported in the caption
        instead of drawn on the panel.
    """
    recall_grid = np.linspace(0, 1, n_points)
    precisions = []

    for yt, yp in zip(y_true_list, y_proba_list):
        precision, recall, _ = precision_recall_curve(yt, yp)
        # Interpolate precision onto common recall grid
        prec_interp = np.interp(recall_grid, recall[::-1], precision[::-1])
        precisions.append(prec_interp)

    precisions = np.array(precisions)
    mean_prec = precisions.mean(axis=0)
    lower = np.percentile(precisions, (100 - ci) / 2, axis=0)
    upper = np.percentile(precisions, 100 - (100 - ci) / 2, axis=0)

    ap_scores = [average_precision_score(yt, yp) for yt, yp in zip(y_true_list, y_proba_list)]
    mean_ap = np.mean(ap_scores)
    std_ap = np.std(ap_scores, ddof=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall_grid, mean_prec, color=color, lw=2.5,
            label=f'{label} (AP = {mean_ap:.3f} ± {std_ap:.3f})')
    ax.fill_between(recall_grid, lower, upper, color=color, alpha=0.2)

    baseline = np.mean(np.concatenate(y_true_list))  # overall prevalence
    if plot_baseline:
        ax.axhline(baseline, color='k', linestyle='--', lw=1, alpha=0.5,
                   label=f'Random classifier (AP = {baseline:.3f})')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return recall_grid, mean_prec, lower, upper, mean_ap, std_ap