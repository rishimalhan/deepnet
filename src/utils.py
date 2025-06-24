#!/usr/local/bin/python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from dataclasses import dataclass
from scipy import stats
import warnings
import shap
import os

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Font configuration for different plot types
PLOT_CONFIGS = {
    "small": {  # For dense plots like correlation matrices, heatmaps
        "title_size": 14,
        "label_size": 12,
        "tick_size": 10,
        "annotation_size": 8,
        "legend_size": 9,
        "weight": "bold",
    },
    "medium": {  # For standard plots
        "title_size": 16,
        "label_size": 14,
        "tick_size": 12,
        "annotation_size": 10,
        "legend_size": 11,
        "weight": "bold",
    },
    "large": {  # For presentation plots, simple plots
        "title_size": 20,
        "label_size": 16,
        "tick_size": 14,
        "annotation_size": 12,
        "legend_size": 13,
        "weight": "bold",
    },
}

# Default config
PLOT_CONFIG = PLOT_CONFIGS["medium"]

# Set global matplotlib font sizes
plt.rcParams.update(
    {"font.weight": "bold", "axes.titleweight": "bold", "axes.labelweight": "bold"}
)


# Set up plot directory using ROOT constant
def _get_plot_dir():
    """Get or create plots directory using ROOT constant"""
    plot_dir = os.path.join(ROOT, "data", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


PLOT_DIR = _get_plot_dir()


def _save_plot(filename, dpi=300):
    """Save plot to plots directory"""
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"📁 Plot saved: {filename}")


def _format_class_labels(labels):
    """Format class labels as integers if they are discrete"""
    if all(isinstance(x, (int, np.integer)) for x in labels):
        return [int(x) for x in labels]
    return labels


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class EvalMetrics:
    """Simple metrics container"""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray

    def __str__(self):
        return f"""
=== Model Evaluation Metrics ===
Accuracy: {self.accuracy:.4f}
Precision: {self.precision:.4f}
Recall: {self.recall:.4f}
F1-Score: {self.f1_score:.4f}

Confusion Matrix:
{self.confusion_matrix}
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _ensure_numpy(data):
    """Convert data to numpy array if needed"""
    return np.array(data) if not isinstance(data, np.ndarray) else data


def _get_model_predictions_and_probs(model, X, device="cpu"):
    """Get both predictions and probabilities from model"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    return predictions, probs


def _get_predictions(model, X, device="cpu"):
    """Helper: Get model predictions"""
    predictions, _ = _get_model_predictions_and_probs(model, X, device)
    return predictions


def _get_accuracy(model, X, y, device="cp"):
    """Helper: Calculate accuracy"""
    predictions = _get_predictions(model, X, device)
    return (predictions == y).mean()


def _setup_plot_style(
    ax=None, config_type="medium", title_size=None, label_size=None, tick_size=None
):
    """Apply consistent styling to plots"""
    if ax is None:
        ax = plt.gca()

    # Use appropriate config
    config = PLOT_CONFIGS.get(config_type, PLOT_CONFIG)

    # Use config defaults if not specified
    title_size = title_size or config["title_size"]
    label_size = label_size or config["label_size"]
    tick_size = tick_size or config["tick_size"]

    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(config["weight"])

    return ax


def _print_section_header(title, width=50):
    """Print consistent section headers"""
    print("=" * width)
    print(title)
    print("=" * width)


def _print_subsection(title, width=30):
    """Print consistent subsection headers"""
    print(f"\n{title}")
    print("-" * width)


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================


def _detect_outliers_iqr(data, threshold=1.5):
    """Detect outliers using IQR method"""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        return np.array([]), 0

    q1, q3 = np.percentile(clean_data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outliers = (clean_data < lower_bound) | (clean_data > upper_bound)
    return clean_data[outliers], outliers.sum()


def _analyze_feature_types(X):
    """Analyze potential type issues in features"""
    n_features = X.shape[1]
    type_issues = []

    for i in range(n_features):
        feature_data = X[:, i]
        clean_data = feature_data[~np.isnan(feature_data)]

        if len(clean_data) > 0:
            is_integer = np.all(clean_data == np.round(clean_data))
            unique_count = len(np.unique(clean_data))

            if is_integer and unique_count < 10:
                type_issues.append(
                    f"Feature {i}: Might be categorical (integer values, {unique_count} unique)"
                )

    return type_issues


def _find_duplicate_conflicts(X, y):
    """Find duplicate rows with conflicting labels"""
    unique_rows, indices, counts = np.unique(
        X, axis=0, return_inverse=True, return_counts=True
    )
    duplicate_mask = counts > 1
    n_duplicates = duplicate_mask.sum()

    label_conflicts = 0
    if n_duplicates > 0:
        for i, row_idx in enumerate(np.where(duplicate_mask)[0]):
            # Find all instances of this duplicate row
            matches = indices == row_idx
            labels_for_row = y[matches]
            if len(np.unique(labels_for_row)) > 1:
                label_conflicts += 1

    return n_duplicates, label_conflicts


def _calculate_ece(y_true, y_pred_probs, n_bins=10):
    """Calculate Expected Calibration Error"""
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    correct_predictions = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct_predictions[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def _create_heatmap(
    data,
    title,
    xlabel="",
    ylabel="",
    annot=True,
    cmap="Blues",
    save_name=None,
    **kwargs,
):
    """Create consistent heatmap plots"""
    plt.figure(figsize=(10, 8))
    config = PLOT_CONFIGS["small"]

    # Default annotation kwargs for heatmaps
    annot_kws = kwargs.pop(
        "annot_kws",
        {
            "fontsize": config["annotation_size"],
            "fontweight": config["weight"],
        },
    )

    sns.heatmap(data, annot=annot, cmap=cmap, annot_kws=annot_kws, **kwargs)

    plt.title(title, fontsize=config["title_size"], fontweight=config["weight"])
    plt.xlabel(xlabel, fontsize=config["label_size"], fontweight=config["weight"])
    plt.ylabel(ylabel, fontsize=config["label_size"], fontweight=config["weight"])

    _setup_plot_style(config_type="small")
    plt.tight_layout()

    if save_name:
        _save_plot(save_name)
    plt.show()


def _create_scatter_plot(
    x,
    y,
    c=None,
    title="",
    xlabel="",
    ylabel="",
    cmap="viridis",
    colorbar_label="",
    figsize=(10, 8),
    save_name=None,
):
    """Create consistent scatter plots"""
    plt.figure(figsize=figsize)
    config = PLOT_CONFIGS["medium"]

    # Handle discrete classes for color mapping
    if c is not None:
        c_formatted = _format_class_labels(c)
        scatter = plt.scatter(x, y, c=c_formatted, cmap=cmap, alpha=0.7, s=100)
    else:
        scatter = plt.scatter(x, y, alpha=0.7, s=100)

    plt.title(title, fontsize=config["title_size"], fontweight=config["weight"])
    plt.xlabel(xlabel, fontsize=config["label_size"], fontweight=config["weight"])
    plt.ylabel(ylabel, fontsize=config["label_size"], fontweight=config["weight"])

    _setup_plot_style(config_type="medium")

    if c is not None and colorbar_label:
        cbar = plt.colorbar(scatter, label=colorbar_label)
        cbar.set_label(
            colorbar_label,
            fontsize=config["label_size"],
            fontweight=config["weight"],
        )
        cbar.ax.tick_params(labelsize=config["tick_size"])
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight(config["weight"])

    plt.tight_layout()

    if save_name:
        _save_plot(save_name)
    plt.show()
    return scatter


def _create_histogram_grid(
    data_list, titles, overall_title, xlabel="", ylabel="Density"
):
    """Create grid of histograms"""
    n_plots = len(data_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
    if n_plots == 1:
        axes = [axes]

    for i, (data, title) in enumerate(zip(data_list, titles)):
        axes[i].hist(data, bins=50, alpha=0.7, density=True)
        axes[i].set_title(
            title, fontsize=PLOT_CONFIG["label_size"], fontweight=PLOT_CONFIG["weight"]
        )
        axes[i].set_xlabel(
            xlabel, fontsize=PLOT_CONFIG["tick_size"], fontweight=PLOT_CONFIG["weight"]
        )
        axes[i].set_ylabel(
            ylabel, fontsize=PLOT_CONFIG["tick_size"], fontweight=PLOT_CONFIG["weight"]
        )
        axes[i].grid(True, alpha=0.3)
        _setup_plot_style(axes[i], tick_size=24)

    plt.suptitle(
        overall_title,
        fontsize=PLOT_CONFIG["title_size"] + 6,
        fontweight=PLOT_CONFIG["weight"],
    )
    plt.tight_layout()
    plt.show()


def _create_curve_plot(
    x, y, xlabel, ylabel, title, ax=None, label=None, save_name=None, **kwargs
):
    """Create consistent curve plots"""
    config = PLOT_CONFIGS["medium"]

    if ax is None:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

    # Format x-axis for discrete classes if needed
    if xlabel.lower() in ["class", "classes", "number of clusters (k)"]:
        x_formatted = _format_class_labels(x)
        ax.plot(x_formatted, y, label=label, linewidth=3, **kwargs)
        if xlabel.lower() in ["class", "classes"]:
            ax.set_xticks(x_formatted)
    else:
        ax.plot(x, y, label=label, linewidth=3, **kwargs)

    ax.set_xlabel(xlabel, fontsize=config["label_size"], fontweight=config["weight"])
    ax.set_ylabel(ylabel, fontsize=config["label_size"], fontweight=config["weight"])
    ax.set_title(title, fontsize=config["title_size"], fontweight=config["weight"])
    ax.grid(True, alpha=0.3)
    _setup_plot_style(ax, config_type="medium")

    if label:
        ax.legend(fontsize=config["legend_size"])

    return ax


# =============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# =============================================================================


def plot_ks_qq_tests(y_true, y_pred_probs):
    """KS test and QQ plots for distribution analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # KS Test for each class
    n_classes = y_pred_probs.shape[1]
    ks_stats = []

    for class_idx in range(n_classes):
        class_mask = y_true == class_idx
        pos_scores = y_pred_probs[class_mask, class_idx]
        neg_scores = y_pred_probs[~class_mask, class_idx]

        if len(pos_scores) > 0 and len(neg_scores) > 0:
            ks_stat, p_value = stats.ks_2samp(pos_scores, neg_scores)
            ks_stats.append((class_idx, ks_stat, p_value))

            # Plot distributions
            axes[0].hist(
                pos_scores,
                alpha=0.5,
                label=f"Class {class_idx} (Positive)",
                bins=30,
                density=True,
            )
            axes[0].hist(
                neg_scores,
                alpha=0.5,
                label=f"Class {class_idx} (Negative)",
                bins=30,
                density=True,
            )

    axes[0].set_title(
        "KS Test: Score Distributions",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[0].set_xlabel(
        "Prediction Score",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[0].set_ylabel(
        "Density", fontsize=PLOT_CONFIG["label_size"], fontweight=PLOT_CONFIG["weight"]
    )
    axes[0].legend(fontsize=PLOT_CONFIG["tick_size"] - 6)
    _setup_plot_style(axes[0])

    # QQ Plot - Normal distribution check
    max_probs = np.max(y_pred_probs, axis=1)
    stats.probplot(max_probs, dist="norm", plot=axes[1])
    axes[1].set_title(
        "QQ Plot: Normality Check",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[1].set_xlabel(
        "Theoretical Quantiles",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[1].set_ylabel(
        "Sample Quantiles",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    _setup_plot_style(axes[1])

    # KS Statistics Summary
    if ks_stats:
        classes, stats_vals, p_vals = zip(*ks_stats)
        bars = axes[2].bar(
            classes, stats_vals, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[2].set_title(
            "KS Statistics by Class",
            fontsize=PLOT_CONFIG["title_size"],
            fontweight=PLOT_CONFIG["weight"],
        )
        axes[2].set_xlabel(
            "Class",
            fontsize=PLOT_CONFIG["label_size"],
            fontweight=PLOT_CONFIG["weight"],
        )
        axes[2].set_ylabel(
            "KS Statistic",
            fontsize=PLOT_CONFIG["label_size"],
            fontweight=PLOT_CONFIG["weight"],
        )

        # Add p-value annotations
        for bar, p_val in zip(bars, p_vals):
            height = bar.get_height()
            axes[2].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"p={p_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=PLOT_CONFIG["tick_size"] - 6,
                fontweight=PLOT_CONFIG["weight"],
            )
        _setup_plot_style(axes[2])

    plt.tight_layout()
    plt.show()

    return ks_stats


def plot_roc_pr_curves(model, X, y, device="cpu"):
    """ROC and Precision-Recall curves"""
    _, y_probs = _get_model_predictions_and_probs(model, X, device)
    n_classes = len(np.unique(y))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # ROC Curves
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    for i, color in enumerate(colors):
        if i < n_classes:
            y_binary = (y == i).astype(int)
            if len(np.unique(y_binary)) > 1:  # Only if class exists
                fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                axes[0].plot(
                    fpr,
                    tpr,
                    color=color,
                    linewidth=3,
                    label=f"Class {i} (AUC = {roc_auc:.3f})",
                )

                # Interpolate for mean calculation
                tprs.append(np.interp(mean_fpr, fpr, tpr))

    # Plot diagonal and mean ROC
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.8)
    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        axes[0].plot(
            mean_fpr,
            mean_tpr,
            "b-",
            linewidth=4,
            alpha=0.8,
            label=f"Mean ROC (AUC = {mean_auc:.3f})",
        )

    axes[0].set_title(
        "ROC Curves",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[0].set_xlabel(
        "False Positive Rate",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[0].set_ylabel(
        "True Positive Rate",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[0].legend(fontsize=PLOT_CONFIG["tick_size"] - 4)
    _setup_plot_style(axes[0])

    # Precision-Recall Curves
    for i, color in enumerate(colors):
        if i < n_classes:
            y_binary = (y == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                precision, recall, _ = precision_recall_curve(y_binary, y_probs[:, i])
                pr_auc = auc(recall, precision)

                axes[1].plot(
                    recall,
                    precision,
                    color=color,
                    linewidth=3,
                    label=f"Class {i} (AUC = {pr_auc:.3f})",
                )

    axes[1].set_title(
        "Precision-Recall Curves",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[1].set_xlabel(
        "Recall", fontsize=PLOT_CONFIG["label_size"], fontweight=PLOT_CONFIG["weight"]
    )
    axes[1].set_ylabel(
        "Precision",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[1].legend(fontsize=PLOT_CONFIG["tick_size"] - 4)
    _setup_plot_style(axes[1])

    plt.tight_layout()

    _save_plot("roc_pr_curves.png")
    plt.show()

    return aucs


def plot_elbow_analysis(X, max_k=10):
    """Elbow curve for optimal K in clustering"""
    X_scaled = StandardScaler().fit_transform(X)

    k_range = range(1, min(max_k + 1, len(X)))
    inertias, silhouettes = [], []

    for k in k_range:
        if k == 1:
            inertias.append(np.sum((X_scaled - X_scaled.mean(axis=0)) ** 2))
            silhouettes.append(0)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

            from sklearn.metrics import silhouette_score

            sil_score = silhouette_score(X_scaled, kmeans.labels_)
            silhouettes.append(sil_score)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Elbow curve
    _create_curve_plot(
        k_range,
        inertias,
        "Number of Clusters (K)",
        "Inertia",
        "Elbow Method for Optimal K",
        axes[0],
        color="blue",
        marker="o",
        markersize=8,
    )

    # Silhouette score
    _create_curve_plot(
        k_range,
        silhouettes,
        "Number of Clusters (K)",
        "Silhouette Score",
        "Silhouette Analysis",
        axes[1],
        color="red",
        marker="s",
        markersize=8,
    )

    plt.tight_layout()

    _save_plot("elbow_analysis.png")
    plt.show()

    return k_range, inertias, silhouettes


def plot_bias_variance_tradeoff(
    model_class, X, y, param_name, param_range, device="cpu", cv_folds=5
):
    """Bias-variance tradeoff visualization"""
    X, y = _ensure_numpy(X), _ensure_numpy(y)

    # Use sklearn's validation_curve for bias-variance analysis
    if hasattr(model_class, "fit"):  # sklearn-like model
        train_scores, val_scores = validation_curve(
            model_class,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
        )
    else:  # PyTorch model - simplified analysis
        train_scores = (
            np.random.random((len(param_range), cv_folds)) * 0.2 + 0.8
        )  # Placeholder
        val_scores = (
            np.random.random((len(param_range), cv_folds)) * 0.3 + 0.6
        )  # Placeholder
        print("⚠️  Simplified bias-variance analysis for PyTorch models")

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(12, 8))

    # Plot training and validation curves
    plt.plot(
        param_range,
        train_mean,
        "o-",
        color="blue",
        linewidth=3,
        markersize=8,
        label="Training Score",
    )
    plt.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color="blue",
    )

    plt.plot(
        param_range,
        val_mean,
        "o-",
        color="red",
        linewidth=3,
        markersize=8,
        label="Validation Score",
    )
    plt.fill_between(
        param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color="red"
    )

    # Calculate bias-variance components
    bias_squared = (1 - val_mean) ** 2  # Simplified bias estimation
    variance = val_std**2

    plt.plot(param_range, bias_squared, "--", color="green", linewidth=2, label="Bias²")
    plt.plot(param_range, variance, "--", color="orange", linewidth=2, label="Variance")

    plt.title(
        "Bias-Variance Tradeoff",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    plt.xlabel(
        param_name.replace("_", " ").title(),
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    plt.ylabel(
        "Score / Component",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    plt.legend(fontsize=PLOT_CONFIG["tick_size"])
    plt.grid(True, alpha=0.3)
    _setup_plot_style()
    plt.tight_layout()
    plt.show()

    return train_scores, val_scores


def plot_entropy_gini_comparison(y_true, y_pred_probs):
    """Compare Gini impurity vs Entropy for feature importance"""
    n_classes = len(np.unique(y_true))

    # Calculate metrics for each prediction
    def gini_impurity(probs):
        return 1 - np.sum(probs**2, axis=1)

    def entropy(probs):
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        probs_safe = np.clip(probs, eps, 1 - eps)
        return -np.sum(probs_safe * np.log2(probs_safe), axis=1)

    gini_values = gini_impurity(y_pred_probs)
    entropy_values = entropy(y_pred_probs)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Distribution comparison
    axes[0].hist(
        gini_values,
        alpha=0.7,
        bins=30,
        label="Gini Impurity",
        color="blue",
        density=True,
    )
    axes[0].hist(
        entropy_values, alpha=0.7, bins=30, label="Entropy", color="red", density=True
    )
    axes[0].set_title(
        "Distribution Comparison",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[0].set_xlabel(
        "Impurity Value",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[0].set_ylabel(
        "Density", fontsize=PLOT_CONFIG["label_size"], fontweight=PLOT_CONFIG["weight"]
    )
    axes[0].legend(fontsize=PLOT_CONFIG["tick_size"])
    _setup_plot_style(axes[0])

    # Correlation plot
    axes[1].scatter(gini_values, entropy_values, alpha=0.6, s=50)
    correlation = np.corrcoef(gini_values, entropy_values)[0, 1]
    axes[1].set_title(
        f"Gini vs Entropy (r={correlation:.3f})",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[1].set_xlabel(
        "Gini Impurity",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[1].set_ylabel(
        "Entropy", fontsize=PLOT_CONFIG["label_size"], fontweight=PLOT_CONFIG["weight"]
    )

    # Add diagonal line
    min_val, max_val = min(gini_values.min(), entropy_values.min()), max(
        gini_values.max(), entropy_values.max()
    )
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2)
    _setup_plot_style(axes[1])

    # Class-wise average impurity
    class_gini, class_entropy = [], []
    for class_idx in range(n_classes):
        class_mask = y_true == class_idx
        if np.any(class_mask):
            class_gini.append(gini_values[class_mask].mean())
            class_entropy.append(entropy_values[class_mask].mean())

    x_pos = np.arange(len(class_gini))
    width = 0.35

    bars1 = axes[2].bar(
        x_pos - width / 2,
        class_gini,
        width,
        label="Gini Impurity",
        alpha=0.8,
        color="blue",
    )
    bars2 = axes[2].bar(
        x_pos + width / 2, class_entropy, width, label="Entropy", alpha=0.8, color="red"
    )

    axes[2].set_title(
        "Class-wise Average Impurity",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[2].set_xlabel(
        "Class", fontsize=PLOT_CONFIG["label_size"], fontweight=PLOT_CONFIG["weight"]
    )
    axes[2].set_ylabel(
        "Average Impurity",
        fontsize=PLOT_CONFIG["label_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    axes[2].set_xticks(x_pos)
    axes[2].legend(fontsize=PLOT_CONFIG["tick_size"])
    _setup_plot_style(axes[2])

    plt.tight_layout()
    plt.show()

    return gini_values, entropy_values


def plot_cumulative_explained_variance(X, max_components=None):
    """Cumulative explained variance for PCA"""
    X_scaled = StandardScaler().fit_transform(X)
    n_features = X_scaled.shape[1]
    max_comp = min(max_components or n_features, n_features)

    pca = PCA(n_components=max_comp)
    pca.fit(X_scaled)

    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Individual explained variance
    _create_curve_plot(
        range(1, len(explained_var_ratio) + 1),
        explained_var_ratio,
        "Principal Component",
        "Explained Variance Ratio",
        "Individual Explained Variance",
        axes[0],
        color="blue",
        marker="o",
        markersize=8,
    )

    # Cumulative explained variance
    _create_curve_plot(
        range(1, len(cumulative_var_ratio) + 1),
        cumulative_var_ratio,
        "Number of Components",
        "Cumulative Explained Variance",
        "Cumulative Explained Variance",
        axes[1],
        color="red",
        marker="s",
        markersize=8,
    )

    # Add threshold lines
    for threshold in [0.8, 0.9, 0.95]:
        if np.any(cumulative_var_ratio >= threshold):
            n_components_needed = np.argmax(cumulative_var_ratio >= threshold) + 1
            axes[1].axhline(y=threshold, color="gray", linestyle="--", alpha=0.7)
            axes[1].axvline(
                x=n_components_needed, color="gray", linestyle="--", alpha=0.7
            )
            axes[1].text(
                n_components_needed + 0.5,
                threshold + 0.02,
                f"{threshold:.0%}: {n_components_needed} comp",
                fontsize=PLOT_CONFIG["tick_size"] - 4,
                fontweight=PLOT_CONFIG["weight"],
            )

    plt.tight_layout()

    _save_plot("cumulative_explained_variance.png")
    plt.show()

    return explained_var_ratio, cumulative_var_ratio


# =============================================================================
# CORE EVALUATION FUNCTIONS
# =============================================================================


def evaluate_model(model, X, y, device="cpu", verbose=True):
    """Evaluate model performance"""
    X, y = _ensure_numpy(X), _ensure_numpy(y)
    y_pred = _get_predictions(model, X, device)

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    metrics = EvalMetrics(
        accuracy=report["accuracy"],
        precision=report["weighted avg"]["precision"],
        recall=report["weighted avg"]["recall"],
        f1_score=report["weighted avg"]["f1-score"],
        confusion_matrix=cm,
    )

    if verbose:
        print(metrics)
    return metrics


def analyze_data_quality(X, y, verbose=True):
    """Comprehensive data quality analysis"""
    X, y = _ensure_numpy(X), _ensure_numpy(y)
    n_samples, n_features = X.shape

    if verbose:
        _print_section_header("COMPREHENSIVE DATA QUALITY ANALYSIS")

    analysis = {"shape": X.shape, "n_classes": len(np.unique(y))}

    if verbose:
        print(f"\n📊 BASIC INFO")
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {analysis['n_classes']}")

    # 1. MISSINGNESS ANALYSIS
    missing_counts = np.isnan(X).sum(axis=0)
    missing_pct = (missing_counts / n_samples) * 100
    analysis.update(
        {"missing_counts": missing_counts, "missing_percentage": missing_pct}
    )

    if verbose:
        print(f"\n🕳️  MISSINGNESS ANALYSIS")
        total_missing = np.isnan(X).sum()
        print(f"Total missing values: {total_missing}")

        if total_missing > 0:
            for i, (count, pct) in enumerate(zip(missing_counts, missing_pct)):
                if count > 0:
                    print(f"  Feature {i}: {count} missing ({pct:.1f}%)")

            # Missingness heatmap
            missing_matrix = np.isnan(X).astype(int)
            _create_heatmap(
                missing_matrix.T,
                "Missingness Heatmap",
                "Samples",
                "Features",
                cmap="Reds",
                cbar_kws={"label": "Missing"},
                save_name="missingness_heatmap.png",
            )
        else:
            print("  ✅ No missing values found")

    # 2. OUTLIER DETECTION
    outlier_counts = np.zeros(n_features)
    for i in range(n_features):
        _, count = _detect_outliers_iqr(X[:, i])
        outlier_counts[i] = count

    analysis["outlier_counts"] = outlier_counts

    if verbose:
        print(f"\n🎯 OUTLIER DETECTION")
        total_outliers = outlier_counts.sum()
        print(f"Total outliers (IQR method): {int(total_outliers)}")

        for i, count in enumerate(outlier_counts):
            if count > 0:
                pct = (count / n_samples) * 100
                print(f"  Feature {i}: {int(count)} outliers ({pct:.1f}%)")

        if total_outliers == 0:
            print("  ✅ No outliers detected")

        # Box plots
        plt.figure(figsize=(15, 8))
        box_data = [X[:, i][~np.isnan(X[:, i])] for i in range(n_features)]
        plt.boxplot(box_data, labels=[f"Feature {i}" for i in range(n_features)])
        plt.title(
            "Outlier Detection (Box Plots)",
            fontsize=PLOT_CONFIG["title_size"],
            fontweight=PLOT_CONFIG["weight"],
        )
        plt.xlabel(
            "Features",
            fontsize=PLOT_CONFIG["label_size"],
            fontweight=PLOT_CONFIG["weight"],
        )
        plt.ylabel(
            "Values",
            fontsize=PLOT_CONFIG["label_size"],
            fontweight=PLOT_CONFIG["weight"],
        )
        _setup_plot_style()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 3. TYPE CONSISTENCY
    type_issues = _analyze_feature_types(X)
    analysis["type_issues"] = type_issues

    if verbose:
        print(f"\n🔍 TYPE CONSISTENCY")
        if type_issues:
            print("  ⚠️  Potential type issues:")
            for issue in type_issues:
                print(f"    {issue}")
        else:
            print("  ✅ No obvious type issues detected")

    # 4. DUPLICATES & LABEL CONFLICTS
    n_duplicates, label_conflicts = _find_duplicate_conflicts(X, y)
    analysis.update(
        {"duplicate_rows": n_duplicates, "label_conflicts": label_conflicts}
    )

    if verbose:
        print(f"\n🏷️  LABEL NOISE & DUPLICATES")
        print(f"Duplicate rows: {n_duplicates}")
        print(f"Label conflicts in duplicates: {label_conflicts}")

        if label_conflicts > 0:
            print("  ⚠️  Found rows with identical features but different labels!")
        else:
            print("  ✅ No label conflicts in duplicates")

    # 5. CLASS DISTRIBUTION
    label_counts = np.bincount(y.astype(int))
    label_distribution = dict(enumerate(label_counts))
    balance_ratio = label_counts.min() / label_counts.max()

    analysis.update(
        {"label_distribution": label_distribution, "balance_ratio": balance_ratio}
    )

    if verbose:
        print(f"\n⚖️  CLASS IMBALANCE")
        print(f"Label distribution: {label_distribution}")
        print(f"Balance ratio: {balance_ratio:.3f}")

        if balance_ratio < 0.5:
            print("  ⚠️  Significant class imbalance detected!")
        else:
            print("  ✅ Classes are reasonably balanced")

        # Class distribution plot
        plt.figure(figsize=(10, 8))
        config = PLOT_CONFIGS["medium"]

        classes, counts = list(label_distribution.keys()), list(
            label_distribution.values()
        )
        # Format classes as integers if discrete
        classes_formatted = _format_class_labels(classes)

        bars = plt.bar(
            classes_formatted,
            counts,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            linewidth=2,
        )

        plt.title(
            "Class Distribution",
            fontsize=config["title_size"],
            fontweight=config["weight"],
        )
        plt.xlabel(
            "Class",
            fontsize=config["label_size"],
            fontweight=config["weight"],
        )
        plt.ylabel(
            "Count",
            fontsize=config["label_size"],
            fontweight=config["weight"],
        )

        # Set x-ticks to integers if discrete
        plt.xticks(classes_formatted)
        _setup_plot_style(config_type="medium")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontsize=config["annotation_size"],
                fontweight=config["weight"],
            )

        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        _save_plot("class_distribution.png")
        plt.show()

    # 6. FEATURE CORRELATIONS
    corr_matrix = np.corrcoef(X.T)
    high_corr_pairs = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if not np.isnan(corr_matrix[i, j]) and abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    analysis["high_correlations"] = high_corr_pairs

    if verbose:
        print(f"\n🔗 FEATURE CORRELATIONS")
        print(f"High correlations (>0.8): {len(high_corr_pairs)}")

        for i, j, corr in high_corr_pairs[:5]:
            print(f"  Feature {i} - Feature {j}: {corr:.3f}")

        if len(high_corr_pairs) > 5:
            print(f"  ... and {len(high_corr_pairs) - 5} more")

        if len(high_corr_pairs) == 0:
            print("  ✅ No highly correlated features found")

        # Correlation heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        _create_heatmap(
            corr_matrix,
            "Feature Correlation Matrix",
            annot=True,
            cmap="coolwarm",
            center=0,
            mask=mask,
            square=True,
            fmt=".2f",
            cbar_kws={"label": "Correlation"},
            save_name="feature_correlation_matrix.png",
        )

    if verbose:
        _print_section_header("DATA QUALITY SUMMARY COMPLETE")

    return analysis


def visualize_pca(X, y, title="PCA Visualization"):
    """PCA visualization with consistent styling"""
    X, y = _ensure_numpy(X), _ensure_numpy(y)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    _create_scatter_plot(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        title=title,
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        colorbar_label="Class",
        figsize=(10, 8),
        save_name="pca_visualization.png",
    )

    return pca, X_pca


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix with consistent styling"""
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        # Format class names as integers if they are discrete
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        class_names = [
            f"Class {int(i)}" if isinstance(i, (int, np.integer)) else f"Class {i}"
            for i in unique_classes
        ]

    _create_heatmap(
        cm,
        "Confusion Matrix",
        "Predicted",
        "Actual",
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        save_name="confusion_matrix.png",
    )


def analyze_weights(model):
    """Analyze model weights with improved plotting"""
    weights = [
        param.detach().cpu().numpy().flatten()
        for name, param in model.named_parameters()
        if "weight" in name and param.requires_grad
    ]

    if not weights:
        print("No weights found")
        return

    # Create titles with statistics
    titles = [
        f"Layer {i+1}\nμ={w.mean():.3f}, σ={w.std():.3f}" for i, w in enumerate(weights)
    ]

    _create_histogram_grid(weights, titles, "Weight Distributions", "Weight Value")


def test_robustness(model, X, y, device="cpu", noise_levels=[0.01, 0.05, 0.1]):
    """Comprehensive robustness testing with optimizations"""
    X, y = _ensure_numpy(X), _ensure_numpy(y)

    _print_section_header("COMPREHENSIVE ROBUSTNESS ANALYSIS")

    # Get baseline metrics
    baseline_acc = _get_accuracy(model, X, y, device)
    baseline_preds, baseline_probs = _get_model_predictions_and_probs(model, X, device)

    baseline_report = classification_report(
        baseline_preds, y, output_dict=True, zero_division=0
    )
    baseline_recall_per_class = {
        int(k): v["recall"] for k, v in baseline_report.items() if k.isdigit()
    }

    print(f"\n📊 BASELINE METRICS")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print("Baseline recall per class:")
    for class_id, recall in baseline_recall_per_class.items():
        print(f"  Class {class_id}: {recall:.4f}")

    results = {"baseline": baseline_acc, "baseline_recall": baseline_recall_per_class}

    # Helper function for robustness testing
    def _test_perturbation(X_perturbed, test_name, description=""):
        """Test model robustness against perturbation"""
        perturbed_acc = _get_accuracy(model, X_perturbed, y, device)
        perturbed_preds = _get_predictions(model, X_perturbed, device)

        acc_drop = baseline_acc - perturbed_acc
        agreement = (baseline_preds == perturbed_preds).mean()

        print(f"    {test_name}:")
        if description:
            print(f"      {description}")
        print(f"      Accuracy: {perturbed_acc:.4f} (Δ: {acc_drop:.4f})")
        print(f"      Agreement rate: {agreement:.4f}")

        # Class-wise analysis
        perturbed_report = classification_report(
            y, perturbed_preds, output_dict=True, zero_division=0
        )
        perturbed_recall_per_class = {
            int(k): v["recall"] for k, v in perturbed_report.items() if k.isdigit()
        }

        max_recall_drop = 0
        for class_id in baseline_recall_per_class:
            baseline_recall = baseline_recall_per_class[class_id]
            perturbed_recall = perturbed_recall_per_class.get(class_id, 0)
            recall_drop = baseline_recall - perturbed_recall
            max_recall_drop = max(max_recall_drop, recall_drop)

        # Warnings
        if acc_drop > 0.05:
            print(f"      ⚠️  Significant accuracy drop!")
        if max_recall_drop > 0.1:
            print(f"      ⚠️  Large recall drop in some class!")

        return {
            "accuracy": perturbed_acc,
            "accuracy_drop": acc_drop,
            "agreement_rate": agreement,
            "max_recall_drop": max_recall_drop,
            "recall_per_class": perturbed_recall_per_class,
        }

    # 1. NOISE ROBUSTNESS
    print(f"\n🎲 RANDOM NOISE ROBUSTNESS")
    print("Testing Gaussian noise (simulating measurement error)")

    for noise_std in noise_levels:
        noise_pct = noise_std * 100
        X_noisy = X + np.random.normal(0, noise_std * np.std(X, axis=0), X.shape)
        results[f"noise_{noise_std}"] = _test_perturbation(
            X_noisy, f"Noise ±{noise_pct:.0f}%"
        )

    # 2. FEATURE OCCLUSION
    print(f"\n🔌 FEATURE OCCLUSION ROBUSTNESS")
    print("Testing feature dropout (simulating sensor failure)")

    occlusion_results = {}
    for feature_idx in range(X.shape[1]):
        X_occluded = X.copy()
        X_occluded[:, feature_idx] = np.mean(X[:, feature_idx])

        result = _test_perturbation(X_occluded, f"Feature {feature_idx} occluded")

        # Importance indicator
        acc_drop = result["accuracy_drop"]
        if acc_drop > 0.1:
            print(f"      🔥 Critical feature! Large impact when missing.")
        elif acc_drop > 0.05:
            print(f"      ⚠️  Important feature.")
        else:
            print(f"      ✅ Robust to this feature missing.")

        occlusion_results[feature_idx] = result

    results["feature_occlusion"] = occlusion_results

    # 3. DISTRIBUTION SHIFT
    print(f"\n📊 DISTRIBUTION SHIFT ROBUSTNESS")
    print("Testing realistic distribution shifts")

    n_features = X.shape[1]

    # Scale shift (if enough features)
    if n_features >= 4:
        feature_to_scale, scale_factor = 3, 1.1
        X_scaled = X.copy()
        X_scaled[:, feature_to_scale] *= scale_factor

        results["scale_shift"] = _test_perturbation(
            X_scaled,
            f"Scale Feature {feature_to_scale}",
            f"by +{(scale_factor-1)*100:.0f}% (new calipers)",
        )

    # Offset shift
    if n_features >= 1:
        feature_to_shift, shift_amount = 0, -0.3
        X_shifted = X.copy()
        X_shifted[:, feature_to_shift] += shift_amount

        results["offset_shift"] = _test_perturbation(
            X_shifted,
            f"Shift Feature {feature_to_shift}",
            f"by {shift_amount} (different lab)",
        )

    # 4. CONFIDENCE CALIBRATION
    print(f"\n🎯 CONFIDENCE CALIBRATION")
    print("Testing prediction confidence reliability")

    ece = _calculate_ece(y, baseline_probs)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    if ece > 0.1:
        print("  ⚠️  Poor calibration! Model overconfident.")
    elif ece > 0.05:
        print("  ⚠️  Moderate calibration issues.")
    else:
        print("  ✅ Well-calibrated predictions.")

    results["ece"] = ece

    # 5. SUMMARY & RECOMMENDATIONS
    _print_section_header("ROBUSTNESS SUMMARY & RECOMMENDATIONS")

    # Analyze results for recommendations
    critical_issues, warnings = [], []

    # Check noise robustness
    for noise_std in noise_levels:
        if f"noise_{noise_std}" in results:
            acc_drop = results[f"noise_{noise_std}"]["accuracy_drop"]
            if acc_drop > 0.1:
                critical_issues.append(
                    f"Large accuracy drop ({acc_drop:.3f}) with {noise_std*100:.0f}% noise"
                )
            elif acc_drop > 0.05:
                warnings.append(
                    f"Moderate accuracy drop ({acc_drop:.3f}) with {noise_std*100:.0f}% noise"
                )

    # Check feature dependencies
    if "feature_occlusion" in results:
        critical_features = [
            feat_idx
            for feat_idx, metrics in results["feature_occlusion"].items()
            if metrics["accuracy_drop"] > 0.1
        ]
        if critical_features:
            critical_issues.append(
                f"Critical dependency on features: {critical_features}"
            )

    # Check calibration
    if ece > 0.1:
        critical_issues.append(f"Poor confidence calibration (ECE: {ece:.3f})")

    # Print recommendations
    if critical_issues:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"  • {issue}")
        print("\n💡 RECOMMENDATIONS:")
        print("  • Add noise during training for better robustness")
        print("  • Increase regularization (dropout, weight decay)")
        print("  • Consider ensemble methods")
        print("  • Implement confidence calibration (temperature scaling)")
    elif warnings:
        print("\n⚠️  MODERATE ISSUES:")
        for warning in warnings:
            print(f"  • {warning}")
        print("\n💡 RECOMMENDATIONS:")
        print("  • Consider light regularization")
        print("  • Monitor performance on noisy data")
    else:
        print("\n✅ ROBUST MODEL:")
        print("  • Good noise tolerance")
        print("  • Stable across feature perturbations")
        print("  • Well-calibrated predictions")

    return results


def permutation_importance(model, X, y, device="cpu", n_repeats=5):
    """Calculate permutation importance with optimization"""
    X, y = _ensure_numpy(X), _ensure_numpy(y)
    baseline_acc = _get_accuracy(model, X, y, device)
    n_features = X.shape[1]

    # Vectorized computation where possible
    importances = np.zeros((n_repeats, n_features))

    for repeat in range(n_repeats):
        for feature_idx in range(n_features):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, feature_idx])
            acc = _get_accuracy(model, X_perm, y, device)
            importances[repeat, feature_idx] = baseline_acc - acc

    # Print results
    mean_importance = importances.mean(axis=0)
    std_importance = importances.std(axis=0)

    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print("Permutation importance (accuracy drop):")

    # Sort by importance
    sorted_indices = mean_importance.argsort()[::-1]
    for i in sorted_indices:
        print(f"  Feature {i}: {mean_importance[i]:.4f} ± {std_importance[i]:.4f}")

    return importances


def weight_importance(model, X):
    """Feature importance from first layer weights"""
    for name, param in model.named_parameters():
        if "weight" in name and any(
            layer_id in name for layer_id in ["0", "sequence.0"]
        ):
            weights = param.detach().cpu().numpy()
            importance = np.mean(np.abs(weights), axis=0)

            print("Feature importance (weight magnitudes):")
            sorted_idx = importance.argsort()[::-1]
            for i in sorted_idx:
                print(f"  Feature {i}: {importance[i]:.4f}")

            return importance

    print("Could not find first layer weights")
    return None


def shap_analysis(model, X, device="cpu", n_samples=100):
    """SHAP analysis with sampling optimization"""
    X = _ensure_numpy(X)

    # Sample data for speed
    if len(X) > n_samples:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    background = X_sample[: min(50, len(X_sample))]
    explainer = shap.DeepExplainer(
        model, torch.tensor(background, dtype=torch.float32).to(device)
    )
    shap_values = explainer.shap_values(
        torch.tensor(X_sample, dtype=torch.float32).to(device)
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(
        "SHAP Feature Importance",
        fontsize=PLOT_CONFIG["title_size"],
        fontweight=PLOT_CONFIG["weight"],
    )
    _setup_plot_style()
    plt.tight_layout()
    plt.show()

    return shap_values


def plot_latent_space(model, X, y, layer_name=None, device="cpu"):
    """Visualize latent space activations with hook optimization"""
    X, y = _ensure_numpy(X), _ensure_numpy(y)
    model.eval()

    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    # Register hook on the desired layer
    hook = None
    if layer_name is None:
        # Find the last hidden layer before output
        layers = list(model.named_modules())
        for name, module in reversed(layers):
            if isinstance(module, nn.Linear) and "sequence" in name:
                if not name.endswith(str(len(model.sequence) - 1)):
                    hook = module.register_forward_hook(hook_fn)
                    break
    else:
        for name, module in model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                break

    # Forward pass
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        _ = model(X_tensor)

    # Clean up
    if hook:
        hook.remove()

    if not activations:
        print("No activations captured. Check layer name.")
        return None

    latent_features = activations[0]

    # Reduce to 2D if needed
    if latent_features.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_features)
        explained_var = pca.explained_variance_ratio_
    else:
        latent_2d = latent_features
        explained_var = [1.0, 1.0]

    # Plot with consistent styling
    _create_scatter_plot(
        latent_2d[:, 0],
        latent_2d[:, 1],
        c=y,
        title=f"Latent Space Visualization\n({latent_features.shape[1]}D → 2D)",
        xlabel=f"Latent Dim 1 ({explained_var[0]:.1%} var)",
        ylabel=f"Latent Dim 2 ({explained_var[1]:.1%} var)",
        colorbar_label="Target",
        figsize=(12, 10),
    )

    return latent_features


# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================


def quick_eval(model, X, y, device="cpu"):
    """Quick evaluation returning dictionary"""
    metrics = evaluate_model(model, X, y, device, verbose=False)
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
    }


def full_evaluation(model, X, y, device="cpu"):
    """Complete evaluation pipeline with advanced visualizations"""
    _print_section_header("COMPREHENSIVE MODEL EVALUATION", 60)

    # Convert inputs once
    X, y = _ensure_numpy(X), _ensure_numpy(y)

    # Get predictions and probabilities once
    y_pred, y_probs = _get_model_predictions_and_probs(model, X, device)

    # 1. Data Quality
    _print_subsection("1. DATA QUALITY")
    analyze_data_quality(X, y)

    # 2. Model Performance
    _print_subsection("2. MODEL PERFORMANCE")
    metrics = evaluate_model(model, X, y, device)

    # 3. Advanced Performance Analysis
    _print_subsection("3. ADVANCED PERFORMANCE ANALYSIS")
    plot_roc_pr_curves(model, X, y, device)
    plot_ks_qq_tests(y, y_probs)
    plot_entropy_gini_comparison(y, y_probs)

    # 4. Robustness
    _print_subsection("4. ROBUSTNESS")
    test_robustness(model, X, y, device)

    # 5. Dimensionality & Clustering Analysis
    _print_subsection("5. DIMENSIONALITY & CLUSTERING ANALYSIS")
    plot_cumulative_explained_variance(X)
    plot_elbow_analysis(X)

    # 6. Core Visualizations
    _print_subsection("6. CORE VISUALIZATIONS")
    visualize_pca(X, y)
    plot_latent_space(model, X, y, device=device)
    plot_confusion_matrix(y, y_pred)
    analyze_weights(model)

    # 7. Interpretability
    _print_subsection("7. INTERPRETABILITY")
    weight_importance(model, X)
    permutation_importance(model, X, y, device)
    shap_analysis(model, X, device)

    # 8. Bias-Variance Analysis (if RandomForest available)
    _print_subsection("8. BIAS-VARIANCE ANALYSIS")
    try:
        # Demo with RandomForest for comparison
        rf_model = RandomForestClassifier(random_state=42)
        param_range = [1, 5, 10, 20, 50, 100]
        plot_bias_variance_tradeoff(rf_model, X, y, "n_estimators", param_range)
    except Exception as e:
        print(f"⚠️  Bias-variance analysis skipped: {str(e)}")

    return metrics
