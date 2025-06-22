#!/usr/local/bin/python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import warnings
import shap

warnings.filterwarnings("ignore")


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


def _get_predictions(model, X, device="cpu"):
    """Helper: Get model predictions"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        return torch.argmax(outputs, dim=1).cpu().numpy()


def _get_accuracy(model, X, y, device="cpu"):
    """Helper: Calculate accuracy"""
    predictions = _get_predictions(model, X, device)
    return (predictions == y).mean()


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def evaluate_model(model, X, y, device="cpu", verbose=True):
    """Evaluate model performance"""
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
    X, y = np.array(X), np.array(y)
    n_samples, n_features = X.shape

    if verbose:
        print("=" * 50)
        print("COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("=" * 50)

    analysis = {}

    # 1. BASIC INFO
    analysis["shape"] = X.shape
    analysis["n_classes"] = len(np.unique(y))

    if verbose:
        print(f"\n📊 BASIC INFO")
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {analysis['n_classes']}")

    # 2. MISSINGNESS ANALYSIS
    missing_counts = np.isnan(X).sum(axis=0)
    missing_pct = (missing_counts / n_samples) * 100
    analysis["missing_counts"] = missing_counts
    analysis["missing_percentage"] = missing_pct

    if verbose:
        print(f"\n🕳️  MISSINGNESS ANALYSIS")
        total_missing = np.isnan(X).sum()
        print(f"Total missing values: {total_missing}")
        if total_missing > 0:
            for i, (count, pct) in enumerate(zip(missing_counts, missing_pct)):
                if count > 0:
                    print(f"  Feature {i}: {count} missing ({pct:.1f}%)")

            # Missingness heatmap
            if total_missing > 0:
                plt.figure(figsize=(12, 8))
                missing_matrix = np.isnan(X).astype(int)
                sns.heatmap(
                    missing_matrix.T, cmap="Reds", cbar_kws={"label": "Missing"}
                )
                plt.title("Missingness Heatmap", fontsize=42, fontweight="bold")
                plt.xlabel("Samples", fontsize=36, fontweight="bold")
                plt.ylabel("Features", fontsize=36, fontweight="bold")
                plt.xticks(fontsize=24, fontweight="bold")
                plt.yticks(fontsize=24, fontweight="bold")
                plt.tight_layout()
                plt.show()
        else:
            print("  ✅ No missing values found")

    # 3. OUTLIER DETECTION
    if verbose:
        print(f"\n🎯 OUTLIER DETECTION")

    outlier_counts = np.zeros(n_features)
    for i in range(n_features):
        feature_data = X[:, i]
        if not np.isnan(feature_data).all():
            # Remove NaN values for outlier detection
            clean_data = feature_data[~np.isnan(feature_data)]
            q1, q3 = np.percentile(clean_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (clean_data < lower_bound) | (clean_data > upper_bound)
            outlier_counts[i] = outliers.sum()

    analysis["outlier_counts"] = outlier_counts

    if verbose:
        total_outliers = outlier_counts.sum()
        print(f"Total outliers (IQR method): {int(total_outliers)}")
        for i, count in enumerate(outlier_counts):
            if count > 0:
                pct = (count / n_samples) * 100
                print(f"  Feature {i}: {int(count)} outliers ({pct:.1f}%)")

        if total_outliers == 0:
            print("  ✅ No outliers detected")

        # Box plots for outlier visualization
        plt.figure(figsize=(15, 8))
        box_data = [X[:, i][~np.isnan(X[:, i])] for i in range(n_features)]
        box_plot = plt.boxplot(
            box_data, labels=[f"Feature {i}" for i in range(n_features)]
        )
        plt.title("Outlier Detection (Box Plots)", fontsize=42, fontweight="bold")
        plt.xlabel("Features", fontsize=36, fontweight="bold")
        plt.ylabel("Values", fontsize=36, fontweight="bold")
        plt.xticks(fontsize=30, fontweight="bold")
        plt.yticks(fontsize=30, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # 4. TYPE CONSISTENCY CHECK
    if verbose:
        print(f"\n🔍 TYPE CONSISTENCY")

    type_issues = []
    for i in range(n_features):
        feature_data = X[:, i]
        clean_data = feature_data[~np.isnan(feature_data)]

        # Check if all values are integers (might be categorical stored as float)
        if len(clean_data) > 0:
            is_integer = np.all(clean_data == np.round(clean_data))
            has_small_range = len(np.unique(clean_data)) < 10

            if is_integer and has_small_range:
                type_issues.append(
                    f"Feature {i}: Might be categorical (integer values, {len(np.unique(clean_data))} unique)"
                )

    analysis["type_issues"] = type_issues

    if verbose:
        if type_issues:
            print("  ⚠️  Potential type issues:")
            for issue in type_issues:
                print(f"    {issue}")
        else:
            print("  ✅ No obvious type issues detected")

    # 5. LABEL NOISE & DUPLICATES
    if verbose:
        print(f"\n🏷️  LABEL NOISE & DUPLICATES")

    # Find duplicate rows
    unique_rows, counts = np.unique(X, axis=0, return_counts=True)
    duplicate_mask = counts > 1
    n_duplicates = duplicate_mask.sum()

    # Check for conflicting labels in duplicates
    label_conflicts = 0
    if n_duplicates > 0:
        for i, row in enumerate(unique_rows[duplicate_mask]):
            # Find all instances of this duplicate row
            matches = np.all(X == row, axis=1)
            labels_for_row = y[matches]
            if len(np.unique(labels_for_row)) > 1:
                label_conflicts += 1

    analysis["duplicate_rows"] = n_duplicates
    analysis["label_conflicts"] = label_conflicts

    if verbose:
        print(f"Duplicate rows: {n_duplicates}")
        print(f"Label conflicts in duplicates: {label_conflicts}")
        if label_conflicts > 0:
            print("  ⚠️  Found rows with identical features but different labels!")
        else:
            print("  ✅ No label conflicts in duplicates")

    # 6. CLASS IMBALANCE
    label_counts = np.bincount(y.astype(int))
    analysis["label_distribution"] = dict(enumerate(label_counts))
    analysis["balance_ratio"] = label_counts.min() / label_counts.max()

    if verbose:
        print(f"\n⚖️  CLASS IMBALANCE")
        print(f"Label distribution: {analysis['label_distribution']}")
        print(f"Balance ratio: {analysis['balance_ratio']:.3f}")

        if analysis["balance_ratio"] < 0.5:
            print("  ⚠️  Significant class imbalance detected!")
        else:
            print("  ✅ Classes are reasonably balanced")

        # Class distribution plot
        plt.figure(figsize=(10, 8))
        classes = list(analysis["label_distribution"].keys())
        counts = list(analysis["label_distribution"].values())
        bars = plt.bar(
            classes, counts, alpha=0.7, color="skyblue", edgecolor="black", linewidth=2
        )
        plt.title("Class Distribution", fontsize=42, fontweight="bold")
        plt.xlabel("Class", fontsize=36, fontweight="bold")
        plt.ylabel("Count", fontsize=36, fontweight="bold")
        plt.xticks(fontsize=30, fontweight="bold")
        plt.yticks(fontsize=30, fontweight="bold")

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontsize=24,
                fontweight="bold",
            )

        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    # 7. FEATURE CORRELATIONS (potential leakage)
    if verbose:
        print(f"\n🔗 FEATURE CORRELATIONS")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    high_corr_pairs = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if not (np.isnan(corr_matrix[i, j])):
                if abs(corr_matrix[i, j]) > 0.8:
                    high_corr_pairs.append((i, j, corr_matrix[i, j]))

    analysis["high_correlations"] = high_corr_pairs

    if verbose:
        print(f"High correlations (>0.8): {len(high_corr_pairs)}")
        for i, j, corr in high_corr_pairs[:5]:  # Show top 5
            print(f"  Feature {i} - Feature {j}: {corr:.3f}")

        if len(high_corr_pairs) > 5:
            print(f"  ... and {len(high_corr_pairs) - 5} more")

        if len(high_corr_pairs) == 0:
            print("  ✅ No highly correlated features found")

        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"label": "Correlation"},
            annot_kws={"fontsize": 20, "fontweight": "bold"},
        )
        plt.title("Feature Correlation Matrix", fontsize=42, fontweight="bold")
        plt.xticks(fontsize=24, fontweight="bold")
        plt.yticks(fontsize=24, fontweight="bold")
        plt.tight_layout()
        plt.show()

    if verbose:
        print(f"\n" + "=" * 50)
        print("DATA QUALITY SUMMARY COMPLETE")
        print("=" * 50)

    return analysis


def visualize_pca(X, y, title="PCA Visualization"):
    """PCA visualization"""
    X, y = np.array(X), np.array(y)

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7, s=100
    )
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=36, fontweight="bold"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=36, fontweight="bold"
    )
    plt.title(title, fontsize=42, fontweight="bold")

    # Increase tick label sizes
    plt.xticks(fontsize=30, fontweight="bold")
    plt.yticks(fontsize=30, fontweight="bold")

    # Colorbar with larger font
    cbar = plt.colorbar(scatter, label="Target")
    cbar.set_label("Target", fontsize=36, fontweight="bold")
    cbar.ax.tick_params(labelsize=30)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.show()

    return pca, X_pca


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 36, "fontweight": "bold"},
    )

    plt.xlabel("Predicted", fontsize=36, fontweight="bold")
    plt.ylabel("Actual", fontsize=36, fontweight="bold")
    plt.title("Confusion Matrix", fontsize=42, fontweight="bold")

    # Increase tick label sizes
    plt.xticks(fontsize=30, fontweight="bold")
    plt.yticks(fontsize=30, fontweight="bold")

    plt.tight_layout()
    plt.show()


def analyze_weights(model):
    """Analyze model weights"""
    weights = [
        param.detach().cpu().numpy().flatten()
        for name, param in model.named_parameters()
        if "weight" in name and param.requires_grad
    ]

    if not weights:
        print("No weights found")
        return

    n_layers = len(weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 8))
    if n_layers == 1:
        axes = [axes]

    for i, w in enumerate(weights):
        axes[i].hist(w, bins=50, alpha=0.7, density=True)
        axes[i].set_title(
            f"Layer {i+1}\nμ={w.mean():.3f}, σ={w.std():.3f}",
            fontsize=36,
            fontweight="bold",
        )
        axes[i].set_xlabel("Weight Value", fontsize=30, fontweight="bold")
        axes[i].set_ylabel("Density", fontsize=30, fontweight="bold")
        axes[i].grid(True, alpha=0.3)

        # Increase tick label sizes
        axes[i].tick_params(axis="both", which="major", labelsize=24)
        for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
            label.set_fontweight("bold")

    plt.suptitle("Weight Distributions", fontsize=48, fontweight="bold")
    plt.tight_layout()
    plt.show()


def test_robustness(model, X, y, device="cpu", noise_levels=[0.01, 0.05, 0.1]):
    """Comprehensive robustness testing"""
    print("=" * 50)
    print("COMPREHENSIVE ROBUSTNESS ANALYSIS")
    print("=" * 50)

    # Get baseline predictions and metrics
    baseline_acc = _get_accuracy(model, X, y, device)
    baseline_preds = _get_predictions(model, X, device)

    # Calculate baseline class-wise recall
    from sklearn.metrics import classification_report

    baseline_report = classification_report(
        y, baseline_preds, output_dict=True, zero_division=0
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

    # 1. RANDOM NOISE ROBUSTNESS
    print(f"\n🎲 RANDOM NOISE ROBUSTNESS")
    print("Testing Gaussian noise (simulating measurement error)")

    for noise_std in noise_levels:
        noise_pct = noise_std * 100
        print(f"\n  Noise level: ±{noise_pct:.0f}%")

        # Add Gaussian noise
        X_noisy = X + np.random.normal(0, noise_std * np.std(X, axis=0), X.shape)

        # Calculate metrics
        noisy_acc = _get_accuracy(model, X_noisy, y, device)
        noisy_preds = _get_predictions(model, X_noisy, device)

        # Accuracy drop
        acc_drop = baseline_acc - noisy_acc
        print(f"    Accuracy: {noisy_acc:.4f} (Δ: {acc_drop:.4f})")

        # Agreement rate
        agreement = (baseline_preds == noisy_preds).mean()
        print(f"    Agreement rate: {agreement:.4f}")

        # Class-wise recall shift
        noisy_report = classification_report(
            y, noisy_preds, output_dict=True, zero_division=0
        )
        noisy_recall_per_class = {
            int(k): v["recall"] for k, v in noisy_report.items() if k.isdigit()
        }

        print(f"    Class-wise recall shifts:")
        max_recall_drop = 0
        for class_id in baseline_recall_per_class:
            baseline_recall = baseline_recall_per_class[class_id]
            noisy_recall = noisy_recall_per_class.get(class_id, 0)
            recall_drop = baseline_recall - noisy_recall
            max_recall_drop = max(max_recall_drop, recall_drop)
            print(f"      Class {class_id}: {noisy_recall:.4f} (Δ: {recall_drop:.4f})")

        # Store results
        results[f"noise_{noise_std}"] = {
            "accuracy": noisy_acc,
            "accuracy_drop": acc_drop,
            "agreement_rate": agreement,
            "max_recall_drop": max_recall_drop,
            "recall_per_class": noisy_recall_per_class,
        }

        # Warning for significant drops
        if acc_drop > 0.05:
            print(
                f"    ⚠️  Significant accuracy drop (>{acc_drop:.3f})! Consider regularization or noise training."
            )
        if max_recall_drop > 0.1:
            print(f"    ⚠️  Large recall drop in some class ({max_recall_drop:.3f})!")

    # 2. FEATURE OCCLUSION / DROPOUT
    print(f"\n🔌 FEATURE OCCLUSION ROBUSTNESS")
    print("Testing feature dropout (simulating sensor failure)")

    n_features = X.shape[1]
    occlusion_results = {}

    for feature_idx in range(n_features):
        print(f"\n  Occluding Feature {feature_idx}:")

        # Create occluded version (set feature to mean value)
        X_occluded = X.copy()
        X_occluded[:, feature_idx] = np.mean(X[:, feature_idx])

        # Calculate metrics
        occluded_acc = _get_accuracy(model, X_occluded, y, device)
        occluded_preds = _get_predictions(model, X_occluded, device)

        # Accuracy drop
        acc_drop = baseline_acc - occluded_acc
        print(f"    Accuracy: {occluded_acc:.4f} (Δ: {acc_drop:.4f})")

        # Agreement rate
        agreement = (baseline_preds == occluded_preds).mean()
        print(f"    Agreement rate: {agreement:.4f}")

        # Class-wise impact
        occluded_report = classification_report(
            y, occluded_preds, output_dict=True, zero_division=0
        )
        occluded_recall_per_class = {
            int(k): v["recall"] for k, v in occluded_report.items() if k.isdigit()
        }

        max_recall_drop = 0
        for class_id in baseline_recall_per_class:
            baseline_recall = baseline_recall_per_class[class_id]
            occluded_recall = occluded_recall_per_class.get(class_id, 0)
            recall_drop = baseline_recall - occluded_recall
            max_recall_drop = max(max_recall_drop, recall_drop)

        occlusion_results[feature_idx] = {
            "accuracy": occluded_acc,
            "accuracy_drop": acc_drop,
            "agreement_rate": agreement,
            "max_recall_drop": max_recall_drop,
        }

        # Feature importance indicator
        if acc_drop > 0.1:
            print(f"    🔥 Critical feature! Large impact when missing.")
        elif acc_drop > 0.05:
            print(f"    ⚠️  Important feature.")
        else:
            print(f"    ✅ Robust to this feature missing.")

    results["feature_occlusion"] = occlusion_results

    # 3. DISTRIBUTION SHIFT ROBUSTNESS
    print(f"\n📊 DISTRIBUTION SHIFT ROBUSTNESS")
    print("Testing realistic distribution shifts")

    # Test 1: Scale specific feature (e.g., petal width +10% - new calipers)
    if n_features >= 4:  # Assuming iris dataset structure
        feature_to_scale = 3  # petal width
        scale_factor = 1.1

        print(
            f"\n  Test 1: Scale Feature {feature_to_scale} by +{(scale_factor-1)*100:.0f}% (new calipers)"
        )
        X_scaled = X.copy()
        X_scaled[:, feature_to_scale] *= scale_factor

        scaled_acc = _get_accuracy(model, X_scaled, y, device)
        scaled_preds = _get_predictions(model, X_scaled, device)
        acc_drop = baseline_acc - scaled_acc
        agreement = (baseline_preds == scaled_preds).mean()

        print(f"    Accuracy: {scaled_acc:.4f} (Δ: {acc_drop:.4f})")
        print(f"    Agreement rate: {agreement:.4f}")

        results["scale_shift"] = {
            "accuracy": scaled_acc,
            "accuracy_drop": acc_drop,
            "agreement_rate": agreement,
        }

    # Test 2: Subtract constant from specific feature (e.g., sepal length -0.3cm - different lab)
    if n_features >= 1:
        feature_to_shift = 0  # sepal length
        shift_amount = -0.3

        print(
            f"\n  Test 2: Shift Feature {feature_to_shift} by {shift_amount} (different lab)"
        )
        X_shifted = X.copy()
        X_shifted[:, feature_to_shift] += shift_amount

        shifted_acc = _get_accuracy(model, X_shifted, y, device)
        shifted_preds = _get_predictions(model, X_shifted, device)
        acc_drop = baseline_acc - shifted_acc
        agreement = (baseline_preds == shifted_preds).mean()

        print(f"    Accuracy: {shifted_acc:.4f} (Δ: {acc_drop:.4f})")
        print(f"    Agreement rate: {agreement:.4f}")

        results["offset_shift"] = {
            "accuracy": shifted_acc,
            "accuracy_drop": acc_drop,
            "agreement_rate": agreement,
        }

    # 4. CONFIDENCE CALIBRATION (ECE approximation)
    print(f"\n🎯 CONFIDENCE CALIBRATION")
    print("Testing prediction confidence reliability")

    # Get prediction probabilities
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Calculate confidence (max probability)
    confidences = np.max(probs, axis=1)
    correct_predictions = baseline_preds == y

    # Simple ECE calculation (10 bins)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find predictions in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct_predictions[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    if ece > 0.1:
        print("  ⚠️  Poor calibration! Model overconfident.")
    elif ece > 0.05:
        print("  ⚠️  Moderate calibration issues.")
    else:
        print("  ✅ Well-calibrated predictions.")

    results["ece"] = ece

    # 5. SUMMARY & RECOMMENDATIONS
    print(f"\n" + "=" * 50)
    print("ROBUSTNESS SUMMARY & RECOMMENDATIONS")
    print("=" * 50)

    # Check for critical issues
    critical_issues = []
    warnings = []

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
        critical_features = []
        for feat_idx, metrics in results["feature_occlusion"].items():
            if metrics["accuracy_drop"] > 0.1:
                critical_features.append(feat_idx)

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
    """Calculate permutation importance"""
    baseline_acc = _get_accuracy(model, X, y, device)
    n_features = X.shape[1]
    importances = np.zeros((n_repeats, n_features))

    for repeat in range(n_repeats):
        for feature_idx in range(n_features):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, feature_idx])
            acc = _get_accuracy(model, X_perm, y, device)
            importances[repeat, feature_idx] = baseline_acc - acc

    # Print results instead of plotting
    mean_importance = importances.mean(axis=0)
    std_importance = importances.std(axis=0)

    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print("Permutation importance (accuracy drop):")
    for i in range(n_features):
        print(f"  Feature {i}: {mean_importance[i]:.4f} ± {std_importance[i]:.4f}")

    return importances


def weight_importance(model, X):
    """Feature importance from first layer weights"""
    for name, param in model.named_parameters():
        if "weight" in name and "0" in name:
            weights = param.detach().cpu().numpy()
            importance = np.mean(np.abs(weights), axis=0)

            # Print results instead of plotting
            print("Feature importance (weight magnitudes):")
            sorted_idx = importance.argsort()[::-1]
            for i in sorted_idx:
                print(f"  Feature {i}: {importance[i]:.4f}")

            return importance

    print("Could not find first layer weights")
    return None


def shap_analysis(model, X, device="cpu", n_samples=100):
    """SHAP analysis (if available)"""
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
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()

    return shap_values


def plot_latent_space(model, X, y, layer_name=None, device="cpu"):
    """Visualize latent space activations"""
    model.eval()

    # Get activations from a specific layer
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    # Register hook on the desired layer (default: last hidden layer)
    if layer_name is None:
        # Find the last layer before output
        layers = list(model.named_modules())
        for name, module in reversed(layers):
            if isinstance(module, nn.Linear) and "sequence" in name:
                # Skip the last layer (output layer)
                if not name.endswith(str(len(model.sequence) - 1)):
                    hook = module.register_forward_hook(hook_fn)
                    break
    else:
        # Register on specific layer
        for name, module in model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                break

    # Forward pass to get activations
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        _ = model(X_tensor)

    # Remove hook
    hook.remove()

    if not activations:
        print("No activations captured. Check layer name.")
        return None

    # Get the activations
    latent_features = activations[0]

    # Reduce to 2D using PCA if needed
    if latent_features.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_features)
        explained_var = pca.explained_variance_ratio_
    else:
        latent_2d = latent_features
        explained_var = [1.0, 1.0]

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        latent_2d[:, 0], latent_2d[:, 1], c=y, cmap="viridis", alpha=0.7, s=100
    )
    plt.xlabel(
        f"Latent Dim 1 ({explained_var[0]:.1%} var)", fontsize=36, fontweight="bold"
    )
    plt.ylabel(
        f"Latent Dim 2 ({explained_var[1]:.1%} var)", fontsize=36, fontweight="bold"
    )
    plt.title(
        f"Latent Space Visualization\n({latent_features.shape[1]}D → 2D)",
        fontsize=42,
        fontweight="bold",
    )

    # Increase tick label sizes
    plt.xticks(fontsize=30, fontweight="bold")
    plt.yticks(fontsize=30, fontweight="bold")

    # Colorbar with larger font
    cbar = plt.colorbar(scatter, label="Target")
    cbar.set_label("Target", fontsize=36, fontweight="bold")
    cbar.ax.tick_params(labelsize=30)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.show()

    return latent_features


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================


def quick_eval(model, X, y, device="cpu"):
    """Quick evaluation"""
    metrics = evaluate_model(model, X, y, device, verbose=False)
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
    }


def full_evaluation(model, X, y, device="cpu"):
    """Complete evaluation pipeline"""
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # 1. Data Quality
    print("\n1. DATA QUALITY")
    print("-" * 30)
    analyze_data_quality(X, y)

    # 2. Model Performance
    print("\n2. MODEL PERFORMANCE")
    print("-" * 30)
    metrics = evaluate_model(model, X, y, device)

    # 3. Robustness
    print("\n3. ROBUSTNESS")
    print("-" * 30)
    test_robustness(model, X, y, device)

    # 4. Visualizations
    print("\n4. VISUALIZATIONS")
    print("-" * 30)
    visualize_pca(X, y)
    plot_latent_space(model, X, y, device=device)
    plot_confusion_matrix(y, _get_predictions(model, X, device))
    analyze_weights(model)

    # 5. Interpretability
    print("\n5. INTERPRETABILITY")
    print("-" * 30)
    weight_importance(model, X)
    permutation_importance(model, X, y, device)

    shap_analysis(model, X, device)

    return metrics


if __name__ == "__main__":
    print("Simplified PyTorch Model Evaluation Utils")
    print("Usage: full_evaluation(model, X, y, device)")
