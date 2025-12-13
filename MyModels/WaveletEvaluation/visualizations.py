"""
Visualization module for model evaluation metrics
Generates publication-quality graphs suitable for academic papers
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os


# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_roc_curve(y_true, y_scores, output_path, eer_threshold=None):
    """
    Plot ROC curve with AUC and EER marked
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted probabilities
        output_path: Path to save the figure
        eer_threshold: EER threshold value to mark on the curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER point
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # Mark EER point
    plt.plot(eer, 1-eer, 'ro', markersize=8, 
             label=f'EER = {eer:.4f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return roc_auc, eer


def plot_precision_recall_curve(y_true, y_scores, output_path):
    """
    Plot Precision-Recall curve with AP score
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        output_path: Path to save the figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {pr_auc:.4f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
             linestyle='--', label=f'Random Classifier (AP = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return pr_auc


def plot_confusion_matrix(y_true, y_pred, output_path, class_names=['Real', 'Fake']):
    """
    Plot confusion matrix as a heatmap
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        output_path: Path to save the figure
        class_names: Names of the classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage'})
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_far_frr_curves(y_true, y_scores, output_path):
    """
    Plot FAR and FRR curves with EER intersection
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        output_path: Path to save the figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # FRR = FNR
    far = fpr  # FAR = FPR
    
    # Find EER
    eer_idx = np.nanargmin(np.absolute(fnr - far))
    eer = far[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far, color='red', lw=2, label='FAR (False Accept Rate)')
    plt.plot(thresholds, fnr, color='blue', lw=2, label='FRR (False Reject Rate)')
    plt.axvline(x=eer_threshold, color='green', linestyle='--', lw=2,
                label=f'EER = {eer:.4f} at threshold = {eer_threshold:.4f}')
    plt.axhline(y=eer, color='green', linestyle='--', lw=1, alpha=0.5)
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Error Rate')
    plt.title('False Accept Rate (FAR) and False Reject Rate (FRR)')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return eer, eer_threshold


def plot_metrics_summary(metrics_dict, output_path):
    """
    Plot summary bar chart of all metrics
    
    Args:
        metrics_dict: Dictionary containing all metrics
        output_path: Path to save the figure
    """
    # Select metrics to display
    display_metrics = {
        'Accuracy': metrics_dict.get('accuracy', 0),
        'Precision': metrics_dict.get('precision', 0),
        'Recall': metrics_dict.get('recall', 0),
        'F1-Score': metrics_dict.get('f1_score', 0),
        'Specificity': metrics_dict.get('specificity', 0),
        'Sensitivity': metrics_dict.get('sensitivity', 0),
        'ROC-AUC': metrics_dict.get('roc_auc', 0),
        'AP': metrics_dict.get('average_precision', 0),
    }
    
    metrics_names = list(display_metrics.keys())
    metrics_values = list(display_metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics Summary')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_score_distribution(y_true, y_scores, output_path):
    """
    Plot distribution of prediction scores for real and fake classes
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        output_path: Path to save the figure
    """
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]
    
    plt.figure(figsize=(10, 6))
    
    # Histograms
    plt.hist(real_scores, bins=50, alpha=0.6, color='blue', 
             label=f'Real (n={len(real_scores)})', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.6, color='red', 
             label=f'Fake (n={len(fake_scores)})', density=True)
    
    # Add vertical line at 0.5 threshold
    plt.axvline(x=0.5, color='black', linestyle='--', lw=2, 
                label='Decision Threshold (0.5)')
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Scores by Class')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics_dict, output_path):
    """
    Plot per-class metrics comparison
    
    Args:
        metrics_dict: Dictionary containing metrics
        output_path: Path to save the figure
    """
    classes = ['Real (Class 0)', 'Fake (Class 1)']
    
    # Extract per-class metrics if available
    real_metrics = {
        'Precision': metrics_dict.get('precision_real', 0),
        'Recall': metrics_dict.get('recall_real', 0),
        'F1-Score': metrics_dict.get('f1_real', 0),
    }
    
    fake_metrics = {
        'Precision': metrics_dict.get('precision_fake', 0),
        'Recall': metrics_dict.get('recall_fake', 0),
        'F1-Score': metrics_dict.get('f1_fake', 0),
    }
    
    metric_names = list(real_metrics.keys())
    real_values = list(real_metrics.values())
    fake_values = list(fake_metrics.values())
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, real_values, width, label='Real', 
                   color='blue', alpha=0.8)
    bars2 = ax.bar(x + width/2, fake_values, width, label='Fake', 
                   color='red', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def create_all_visualizations(y_true, y_scores, y_pred, metrics_dict, output_dir):
    """
    Create all visualizations and save to output directory
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        y_pred: Predicted binary labels
        metrics_dict: Dictionary of all calculated metrics
        output_dir: Directory to save all visualizations
    
    Returns:
        Dictionary with paths to all generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    figure_paths = {}
    
    print("Generating visualizations...")
    
    # ROC Curve
    print("  - ROC Curve...")
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(y_true, y_scores, roc_path)
    figure_paths['roc_curve'] = roc_path
    
    # Precision-Recall Curve
    print("  - Precision-Recall Curve...")
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(y_true, y_scores, pr_path)
    figure_paths['pr_curve'] = pr_path
    
    # Confusion Matrix
    print("  - Confusion Matrix...")
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path)
    figure_paths['confusion_matrix'] = cm_path
    
    # FAR/FRR Curves
    print("  - FAR/FRR Curves...")
    far_frr_path = os.path.join(output_dir, 'far_frr_curves.png')
    plot_far_frr_curves(y_true, y_scores, far_frr_path)
    figure_paths['far_frr'] = far_frr_path
    
    # Metrics Summary
    print("  - Metrics Summary...")
    summary_path = os.path.join(output_dir, 'metrics_summary.png')
    plot_metrics_summary(metrics_dict, summary_path)
    figure_paths['metrics_summary'] = summary_path
    
    # Score Distribution
    print("  - Score Distribution...")
    dist_path = os.path.join(output_dir, 'score_distribution.png')
    plot_score_distribution(y_true, y_scores, dist_path)
    figure_paths['score_distribution'] = dist_path
    
    # Per-class Metrics
    print("  - Per-Class Metrics...")
    per_class_path = os.path.join(output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(metrics_dict, per_class_path)
    figure_paths['per_class_metrics'] = per_class_path
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return figure_paths
