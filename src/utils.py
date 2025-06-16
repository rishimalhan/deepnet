#!/usr/local/bin/python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Optional imports
try:
    from mlxtend.plotting import plot_decision_regions

    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def to_numpy(
    data: Union[torch.Tensor, pd.DataFrame, pd.Series, np.ndarray],
) -> np.ndarray:
    """Convert various data types to numpy array"""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    return np.asarray(data)


def setup_plot(
    figsize: Tuple[int, int] = (10, 8), title: str = "", save_path: Optional[str] = None
) -> plt.Figure:
    """Setup matplotlib figure with common settings"""
    fig = plt.figure(figsize=figsize)
    if title:
        plt.suptitle(title, fontsize=14, y=0.98)
    return fig


def save_and_show(save_path: Optional[str] = None):
    """Save and show plot with common settings"""
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# CORE EVALUATION FUNCTIONS
# =============================================================================


@dataclass
class EvalMetrics:
    """Simplified metrics dataclass using sklearn's classification_report"""

    report_dict: Dict[str, Any]
    confusion_matrix: np.ndarray

    @property
    def accuracy(self) -> float:
        return self.report_dict["accuracy"]

    @property
    def weighted_avg(self) -> Dict[str, float]:
        return self.report_dict["weighted avg"]

    @property
    def macro_avg(self) -> Dict[str, float]:
        return self.report_dict["macro avg"]

    def __str__(self) -> str:
        from sklearn.metrics import classification_report as cr

        return f"""
=== Model Evaluation Metrics ===
Accuracy: {self.accuracy:.4f}

Macro Averages:
  Precision: {self.macro_avg['precision']:.4f}
  Recall: {self.macro_avg['recall']:.4f}
  F1-Score: {self.macro_avg['f1-score']:.4f}

Weighted Averages:
  Precision: {self.weighted_avg['precision']:.4f}
  Recall: {self.weighted_avg['recall']:.4f}
  F1-Score: {self.weighted_avg['f1-score']:.4f}

Confusion Matrix:
{self.confusion_matrix}
"""


def evaluate_model(
    model: nn.Module,
    dataset,
    device: str = "cpu",
    batch_size: int = 64,
    verbose: bool = True,
) -> EvalMetrics:
    """Streamlined model evaluation using sklearn"""
    model.eval()
    model = model.to(device)

    # Create dataloader if needed
    if not isinstance(dataset, DataLoader):
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dataset:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            if outputs.dim() > 1 and outputs.size(1) > 1:
                predictions = torch.argmax(outputs, dim=1)
            else:
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).long()

            y_true.extend(to_numpy(targets))
            y_pred.extend(to_numpy(predictions))

    # Use sklearn for all metrics
    report_dict = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = EvalMetrics(report_dict, cm)

    if verbose:
        print(metrics)

    return metrics


# =============================================================================
# DATA ANALYSIS FUNCTIONS
# =============================================================================


def analyze_data_quality(
    X, y, feature_names: Optional[List[str]] = None, verbose: bool = True
) -> Dict[str, Any]:
    """Simplified data quality analysis"""
    X, y = to_numpy(X), to_numpy(y)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names)

    # Core analysis using pandas
    analysis = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "correlations": df.corr(),
        "statistics": df.describe(),
        "label_distribution": pd.Series(y).value_counts().to_dict(),
        "balance_ratio": pd.Series(y).value_counts().min()
        / pd.Series(y).value_counts().max(),
    }

    # High correlations
    corr_matrix = analysis["correlations"]
    high_corr = np.where(np.triu(np.abs(corr_matrix), k=1) > 0.8)
    analysis["high_correlations"] = [
        (feature_names[i], feature_names[j], corr_matrix.iloc[i, j])
        for i, j in zip(*high_corr)
    ]

    if verbose:
        print(f"Dataset shape: {analysis['shape']}")
        print(f"Label distribution: {analysis['label_distribution']}")
        print(f"Balance ratio: {analysis['balance_ratio']:.3f}")
        print(f"High correlations (>0.8): {len(analysis['high_correlations'])}")
        if analysis["high_correlations"]:
            for feat1, feat2, corr in analysis["high_correlations"][:5]:
                print(f"  {feat1} - {feat2}: {corr:.3f}")

    return analysis


def visualize_pca(
    X,
    y,
    n_components: int = 2,
    title: str = "PCA Visualization",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """Simplified PCA visualization"""
    X, y = to_numpy(X), to_numpy(y)

    # Standardize and apply PCA
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    setup_plot(figsize, title)

    if n_components == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.colorbar(label="Target")
    elif n_components == 3:
        ax = plt.axes(projection="3d")
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap="viridis")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")

    save_and_show(save_path)
    return pca, X_pca


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """Simplified confusion matrix plot"""
    cm = confusion_matrix(to_numpy(y_true), to_numpy(y_pred))

    setup_plot(figsize, "Confusion Matrix")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_and_show(save_path)


def plot_decision_boundaries(
    X,
    y,
    model,
    feature_indices: Tuple[int, int] = (0, 1),
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """Simplified decision boundary plotting"""
    if not MLXTEND_AVAILABLE:
        print("Install mlxtend: pip install mlxtend")
        return

    X, y = to_numpy(X), to_numpy(y)
    X_subset = X[:, list(feature_indices)]

    setup_plot(figsize, "Decision Boundaries")
    plot_decision_regions(X_subset, y, clf=model, legend=2)
    plt.xlabel(f"Feature {feature_indices[0]}")
    plt.ylabel(f"Feature {feature_indices[1]}")
    save_and_show(save_path)


# =============================================================================
# ROBUSTNESS TESTING
# =============================================================================


def test_robustness(
    model: nn.Module,
    X,
    y,
    device: str = "cpu",
    noise_levels: List[float] = [0.01, 0.05, 0.1],
    verbose: bool = True,
) -> Dict[str, float]:
    """Simplified robustness testing using vectorized operations"""
    model.eval()
    X, y = torch.tensor(to_numpy(X), dtype=torch.float32), torch.tensor(
        to_numpy(y), dtype=torch.long
    )

    # Baseline accuracy
    with torch.no_grad():
        outputs = model(X.to(device))
        if outputs.dim() > 1 and outputs.size(1) > 1:
            baseline_pred = torch.argmax(outputs, dim=1)
        else:
            baseline_pred = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
        baseline_acc = (baseline_pred.cpu() == y).float().mean().item()

    results = {"baseline": baseline_acc}

    if verbose:
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        print("Noise robustness:")

    # Test noise robustness
    for noise_std in noise_levels:
        noise = torch.randn_like(X) * noise_std
        X_noisy = X + noise

        with torch.no_grad():
            outputs = model(X_noisy.to(device))
            if outputs.dim() > 1 and outputs.size(1) > 1:
                noisy_pred = torch.argmax(outputs, dim=1)
            else:
                noisy_pred = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
            noisy_acc = (noisy_pred.cpu() == y).float().mean().item()

        results[f"noise_{noise_std}"] = noisy_acc
        if verbose:
            print(
                f"  σ={noise_std}: {noisy_acc:.4f} (drop: {baseline_acc-noisy_acc:.4f})"
            )

    return results


# =============================================================================
# WEIGHT ANALYSIS
# =============================================================================


def analyze_weights(
    model: nn.Module,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None,
):
    """Simplified weight analysis"""
    weights_data = {}

    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weights_data[name] = to_numpy(param).flatten()

    if not weights_data:
        print("No weights found")
        return

    n_layers = len(weights_data)
    cols = min(4, n_layers)
    rows = (n_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes

    for i, (name, weights) in enumerate(weights_data.items()):
        ax = axes[i] if n_layers > 1 else axes
        ax.hist(weights, bins=50, alpha=0.7, density=True)
        ax.set_title(
            f'{name.split(".")[-2]}\n μ={weights.mean():.3f}, σ={weights.std():.3f}'
        )
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Weight Distributions", fontsize=16)
    save_and_show(save_path)


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================


def preprocess_features(X, method: str = "standard", scaler=None):
    """Unified preprocessing function"""
    X_np = to_numpy(X)

    if scaler is None:
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        X_scaled = scaler.fit_transform(X_np)
    else:
        X_scaled = scaler.transform(X_np)

    # Return in same format as input
    if isinstance(X, torch.Tensor):
        return torch.tensor(X_scaled, dtype=X.dtype), scaler
    elif isinstance(X, pd.DataFrame):
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler
    return X_scaled, scaler


# =============================================================================
# COMPREHENSIVE EVALUATION PIPELINE
# =============================================================================


def full_evaluation(
    model: nn.Module,
    X_test,
    y_test,
    feature_names: Optional[List[str]] = None,
    device: str = "cpu",
    figsize: Tuple[int, int] = (15, 12),
):
    """Complete evaluation pipeline in one function"""
    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # 1. Data Quality
    print("\n1. DATA QUALITY ANALYSIS")
    print("-" * 30)
    data_quality = analyze_data_quality(X_test, y_test, feature_names, verbose=True)

    # 2. Model Performance
    print("\n2. MODEL PERFORMANCE")
    print("-" * 30)
    # Convert to dataset format for model evaluation
    dataset = list(
        zip(
            torch.tensor(to_numpy(X_test), dtype=torch.float32),
            torch.tensor(to_numpy(y_test), dtype=torch.long),
        )
    )
    metrics = evaluate_model(model, dataset, device, verbose=True)

    # 3. Robustness
    print("\n3. ROBUSTNESS ANALYSIS")
    print("-" * 30)
    robustness = test_robustness(model, X_test, y_test, device, verbose=True)

    # 4. Visualizations
    print("\n4. GENERATING VISUALIZATIONS...")
    print("-" * 30)

    fig = plt.figure(figsize=figsize)

    # PCA plot
    plt.subplot(2, 3, 1)
    pca, X_pca = visualize_pca(X_test, y_test, title="PCA Analysis", save_path=None)
    plt.subplot(2, 3, 2)
    plot_confusion_matrix(
        y_test,
        [metrics.confusion_matrix.argmax(axis=1)[i] for i in range(len(y_test))],
        save_path=None,
    )

    # Weight analysis
    plt.subplot(2, 3, (3, 6))
    analyze_weights(model, save_path=None)

    plt.tight_layout()
    plt.show()

    return {
        "data_quality": data_quality,
        "metrics": metrics,
        "robustness": robustness,
        "pca": (pca, X_pca),
    }


# =============================================================================
# QUICK ACCESS FUNCTIONS
# =============================================================================


def quick_eval(model, X_test, y_test, device="cpu") -> Dict[str, float]:
    """Quick evaluation returning key metrics"""
    dataset = list(
        zip(
            torch.tensor(to_numpy(X_test), dtype=torch.float32),
            torch.tensor(to_numpy(y_test), dtype=torch.long),
        )
    )
    metrics = evaluate_model(model, dataset, device, verbose=False)

    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.macro_avg["precision"],
        "recall": metrics.macro_avg["recall"],
        "f1_score": metrics.macro_avg["f1-score"],
    }


if __name__ == "__main__":
    print("Optimized PyTorch Model Evaluation Utils")
    print("Key functions: evaluate_model, analyze_data_quality, visualize_pca")
    print("Quick eval: quick_eval(model, X_test, y_test)")
    print("Full pipeline: full_evaluation(model, X_test, y_test)")
