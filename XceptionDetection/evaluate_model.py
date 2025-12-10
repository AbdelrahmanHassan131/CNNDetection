"""
Comprehensive Model Evaluation Script for Xception Deepfake Detection
Loads trained model, runs inference on validation dataset, and generates publication-quality metrics and visualizations
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_curve
)

from networks.trainer import Trainer
from data import create_dataloader
from visualizations import create_all_visualizations


class EvaluationOptions:
    """Options for evaluation"""
    def __init__(self, dataroot, checkpoint_path, checkpoint_dir, name, epoch='latest'):
        self.dataroot = dataroot
        self.checkpoint_path = checkpoint_path
        self.checkpoints_dir = checkpoint_dir
        self.name = name
        self.epoch = epoch
        
        # Data loading options - fake=0, real=1
        self.classes = ['fake', 'real']
        self.batch_size = 32
        self.loadSize = 333  # Xception: resize to 333
        self.cropSize = 299  # Xception: center crop to 299
        self.num_threads = 4
        
        # Evaluation mode settings
        self.isTrain = False
        self.no_resize = False
        self.no_crop = False
        self.no_flip = True
        self.serial_batches = True
        self.class_bal = False
        
        # GPU settings
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        
        # Model settings (needed for loading)
        self.continue_train = False
        self.init_gain = 0.02




def calculate_eer(y_true, y_scores):
    """
    Calculate Equal Error Rate (EER)
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
    
    Returns:
        eer: Equal Error Rate
        eer_threshold: Threshold at which EER occurs
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    
    # Find the threshold where FPR and FNR are closest
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold


def calculate_far_frr(y_true, y_scores, threshold=0.5):
    """
    Calculate False Accept Rate (FAR) and False Reject Rate (FRR)
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        threshold: Decision threshold
    
    Returns:
        far: False Accept Rate
        frr: False Reject Rate
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    # FAR: False positives / Total negatives (fake images classified as real)
    negatives = y_true == 0
    if negatives.sum() > 0:
        far = ((y_pred == 1) & (y_true == 0)).sum() / negatives.sum()
    else:
        far = 0.0
    
    # FRR: False negatives / Total positives (real images classified as fake)
    positives = y_true == 1
    if positives.sum() > 0:
        frr = ((y_pred == 0) & (y_true == 1)).sum() / positives.sum()
    else:
        frr = 0.0
    
    return far, frr


def calculate_all_metrics(y_true, y_scores, threshold=0.5):
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: Ground truth labels (0=fake, 1=real)
        y_scores: Predicted probabilities
        threshold: Decision threshold for binary classification
    
    Returns:
        Dictionary containing all metrics
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except:
        roc_auc = 0.0
    
    # Average Precision (AP)
    try:
        ap = average_precision_score(y_true, y_scores)
    except:
        ap = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Specificity and Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    
    # EER
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    
    # FAR and FRR at default threshold
    far, frr = calculate_far_frr(y_true, y_scores, threshold)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    
    # Per-class AUC (one-vs-rest)
    try:
        # AUC for class 0 (Fake): probability of being fake
        auc_fake = roc_auc_score(y_true == 0, y_scores)
    except:
        auc_fake = 0.0
    
    try:
        # AUC for class 1 (Real): probability of being real
        y_scores_real = 1 - y_scores
        auc_real = roc_auc_score(y_true == 1, y_scores_real)
    except:
        auc_real = 0.0
    
    metrics = {
        # Basic metrics
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        
        # Advanced metrics
        'roc_auc': float(roc_auc),
        'average_precision': float(ap),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        
        # Confusion matrix components
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        
        # Specificity and Sensitivity
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        
        # FAR and FRR
        'far': float(far),
        'frr': float(frr),
        'decision_threshold': float(threshold),
        
        # Per-class metrics
        'precision_real': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        'precision_fake': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        'recall_real': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        'recall_fake': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
        'f1_real': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        'f1_fake': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        'support_real': int(support[0]) if len(support) > 0 else 0,
        'support_fake': int(support[1]) if len(support) > 1 else 0,
        
        # Per-class AUC
        'auc_real': float(auc_real),
        'auc_fake': float(auc_fake),
        
        # Dataset info
        'total_samples': len(y_true),
        'num_fake': int((y_true == 0).sum()),
        'num_real': int((y_true == 1).sum()),
    }
    
    return metrics


def run_evaluation(model, data_loader, device):
    """
    Run model inference on the entire dataset
    
    Args:
        model: The trained model
        data_loader: DataLoader for validation data
        device: Device to run inference on
    
    Returns:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
    """
    model.eval()
    
    y_true = []
    y_scores = []
    
    print("\nRunning inference on validation dataset...")
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            images, labels = data
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Handle single sample case
            if probs.ndim == 0:
                probs = np.array([float(probs)])
            
            y_scores.extend(probs.flatten().tolist())
            y_true.extend(labels.numpy().flatten().tolist())
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    return y_true, y_scores


def print_metrics_summary(metrics):
    """Print a formatted summary of all metrics"""
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    
    print(f"\nDataset Information:")
    print(f"  Total Samples: {metrics['total_samples']}")
    print(f"  Real Images: {metrics['num_real']}")
    print(f"  Fake Images: {metrics['num_fake']}")
    
    print(f"\nBasic Classification Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    
    print(f"\nAdvanced Metrics:")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  AP (Average Precision): {metrics['average_precision']:.4f}")
    print(f"  EER (Equal Error Rate): {metrics['eer']:.4f} (at threshold {metrics['eer_threshold']:.4f})")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    
    print(f"\nSpecificity & Sensitivity:")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    
    print(f"\nFAR & FRR (at threshold {metrics['decision_threshold']}):")
    print(f"  FAR (False Accept Rate):  {metrics['far']:.4f}")
    print(f"  FRR (False Reject Rate):  {metrics['frr']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Fake - Precision: {metrics['precision_fake']:.4f}, Recall: {metrics['recall_fake']:.4f}, F1: {metrics['f1_fake']:.4f}, AUC: {metrics['auc_fake']:.4f}")
    print(f"  Real - Precision: {metrics['precision_real']:.4f}, Recall: {metrics['recall_real']:.4f}, F1: {metrics['f1_real']:.4f}, AUC: {metrics['auc_real']:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Xception model on validation dataset')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to validation dataset (should contain fake and real folders)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file (e.g., latest_net.pth)')
    parser.add_argument('--name', type=str, default='xception_experiment',
                       help='Experiment name (used to locate checkpoint if not absolute path)')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results and visualizations')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold for binary classification')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup options
    print("Setting up evaluation options...")
    
    # Determine checkpoint directory and epoch
    if os.path.isfile(args.checkpoint):
        # Absolute path to checkpoint file
        checkpoint_path = args.checkpoint
        checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        epoch = os.path.basename(checkpoint_path).replace('_net.pth', '')
    else:
        # Relative path or epoch name
        checkpoint_dir = args.checkpoints_dir
        epoch = args.checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, args.name, f'{epoch}_net.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    opt = EvaluationOptions(
        dataroot=args.dataroot,
        checkpoint_path=checkpoint_path,
        checkpoint_dir=checkpoint_dir,
        name=args.name,
        epoch=epoch
    )
    opt.batch_size = args.batch_size
    opt.gpu_ids = [args.gpu_id] if args.gpu_id >= 0 and torch.cuda.is_available() else []
    
    # Load model
    print("Loading model...")
    model = Trainer(opt)
    device = model.device
    
    # Load checkpoint directly from the provided path
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint is a dict with 'model' key or direct state_dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Loaded model from checkpoint dictionary")
        else:
            state_dict = checkpoint
            print("Loaded model state_dict directly")
        
        model.model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")
    
    # Create data loader
    print(f"Loading validation data from: {args.dataroot}")
    data_loader = create_dataloader(opt)
    print(f"Total batches: {len(data_loader)}")
    
    # Run evaluation
    y_true, y_scores = run_evaluation(model.model, data_loader, device)
    
    print(f"\nCollected {len(y_true)} predictions")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_all_metrics(y_true, y_scores, threshold=args.threshold)
    
    # Print summary
    print_metrics_summary(metrics)
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    y_pred = (y_scores >= args.threshold).astype(int)
    figure_paths = create_all_visualizations(
        y_true, y_scores, y_pred, metrics, args.output_dir
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - metrics.json: All calculated metrics")
    print(f"  - roc_curve.png: ROC curve with AUC and EER")
    print(f"  - precision_recall_curve.png: Precision-Recall curve with AP")
    print(f"  - confusion_matrix.png: Confusion matrix (counts and normalized)")
    print(f"  - far_frr_curves.png: FAR and FRR curves with EER")
    print(f"  - metrics_summary.png: Bar chart of all metrics")
    print(f"  - score_distribution.png: Distribution of prediction scores")
    print(f"  - per_class_metrics.png: Per-class performance comparison")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
