"""
Comprehensive Evaluation Script for MHA_128 Fusion Model
Multi-Head Attention fusion of RGB (Wang2020) and Wavelet (Wolter) models
"""

import os
import sys
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

# Add MyModels to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.MHA_128.Trainer_MHA_128 import MHAFusionTrainer
from visualizations import create_all_visualizations


class EvaluationOptions:
    """Options for evaluation"""
    def __init__(self, dataroot, checkpoint_path, rgb_model_path, wavelet_model_path, 
                 checkpoint_dir, name, epoch='latest'):
        self.dataroot = dataroot
        self.checkpoint_path = checkpoint_path
        self.rgb_model_path = rgb_model_path
        self.wavelet_model_path = wavelet_model_path
        self.checkpoints_dir = checkpoint_dir
        self.name = name
        self.epoch = epoch
        
        # Data loading options
        self.classes = ['fake', 'real']
        self.batch_size = 32
        self.loadSize = 256  # Match training: resize to 256
        self.cropSize = 224  # Match training: center crop to 224
        self.num_threads = 0
        
        # Wavelet parameters
        self.wavelet_type = 'haar'
        self.wavelet_level = 3
        self.wavelet_mode = 'reflect'
        self.use_log_packets = True
        
        # MHA fusion parameters
        self.embed_dim = 128
        self.num_heads = 4
        self.dropout = 0.1
        self.fusion_type = 'cross_attention'
        self.freeze_base_models = True
        
        # Evaluation mode settings
        self.isTrain = False
        self.no_resize = False
        self.no_crop = False
        self.no_flip = True
        self.serial_batches = True
        self.class_bal = False
        self.mode = 'binary'
        
        # Data augmentation (disabled for evaluation)
        self.blur_prob = 0.0
        self.blur_sig = [0.0]
        self.jpg_prob = 0.0
        self.jpg_method = ['cv2']
        self.jpg_qual = [100]
        self.rz_interp = ['bilinear']
        
        # GPU settings
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        
        # Model settings
        self.continue_train = False
        self.init_gain = 0.02
        self.optim = 'adam'
        self.lr = 0.0001
        self.beta1 = 0.9


def calculate_eer(y_true, y_scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold


def calculate_far_frr(y_true, y_scores, threshold=0.5):
    """Calculate FAR and FRR"""
    y_pred = (y_scores >= threshold).astype(int)
    
    negatives = y_true == 0
    if negatives.sum() > 0:
        far = ((y_pred == 1) & (y_true == 0)).sum() / negatives.sum()
    else:
        far = 0.0
    
    positives = y_true == 1
    if positives.sum() > 0:
        frr = ((y_pred == 0) & (y_true == 1)).sum() / positives.sum()
    else:
        frr = 0.0
    
    return far, frr


def calculate_all_metrics(y_true, y_scores, threshold=0.5):
    """Calculate all evaluation metrics"""
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
    
    # Average Precision
    try:
        ap = average_precision_score(y_true, y_scores)
    except:
        ap = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Specificity and Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # EER
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    
    # FAR and FRR
    far, frr = calculate_far_frr(y_true, y_scores, threshold)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    
    # Per-class AUC
    try:
        auc_fake = roc_auc_score(y_true == 0, y_scores)
    except:
        auc_fake = 0.0
    
    try:
        y_scores_real = 1 - y_scores
        auc_real = roc_auc_score(y_true == 1, y_scores_real)
    except:
        auc_real = 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'average_precision': float(ap),
        'eer': float(eer),
        'eer_threshold': float(eer_threshold),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'far': float(far),
        'frr': float(frr),
        'decision_threshold': float(threshold),
        'precision_fake': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        'precision_real': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        'recall_fake': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        'recall_real': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
        'f1_fake': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        'f1_real': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        'support_fake': int(support[0]) if len(support) > 0 else 0,
        'support_real': int(support[1]) if len(support) > 1 else 0,
        'auc_real': float(auc_real),
        'auc_fake': float(auc_fake),
        'total_samples': len(y_true),
        'num_fake': int((y_true == 0).sum()),
        'num_real': int((y_true == 1).sum()),
    }
    
    return metrics


def run_evaluation(model, data_loader, device):
    """Run model inference on validation dataset"""
    model.eval()
    
    y_true = []
    y_scores = []
    
    print("\nRunning inference on validation dataset...")
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            # MHA dual-input: (rgb_images, wavelet_packets, labels)
            rgb_images, wavelet_packets, labels = data
            rgb_images = rgb_images.to(device)
            wavelet_packets = wavelet_packets.to(device)
            
            # Forward pass through fusion model
            outputs = model(rgb_images, wavelet_packets)
            
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
    """Print formatted metrics summary"""
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY - MHA_128 Fusion")
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
    parser = argparse.ArgumentParser(description='Evaluate MHA_128 fusion model')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to validation dataset (containing fake and real folders)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to MHA fusion model checkpoint file')
    parser.add_argument('--rgb_model_path', type=str, required=True,
                       help='Path to pre-trained wang2020_128 checkpoint')
    parser.add_argument('--wavelet_model_path', type=str, required=True,
                       help='Path to pre-trained WaveletsPacketsScratch_128 checkpoint')
    parser.add_argument('--name', type=str, default='mha_experiment',
                       help='Experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (-1 for CPU)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Setting up evaluation options...")
    
    # Determine checkpoint path
    if os.path.isfile(args.checkpoint):
        checkpoint_path = args.checkpoint
        checkpoint_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        epoch = os.path.basename(checkpoint_path).replace('_net.pth', '').replace('.pth', '')
    else:
        checkpoint_dir = args.checkpoints_dir
        epoch = args.checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, args.name, f'{epoch}_net.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Verify base model paths exist
    if not os.path.exists(args.rgb_model_path):
        print(f"Error: RGB model not found at {args.rgb_model_path}")
        return
    
    if not os.path.exists(args.wavelet_model_path):
        print(f"Error: Wavelet model not found at {args.wavelet_model_path}")
        return
    
    print(f"Loading MHA fusion checkpoint from: {checkpoint_path}")
    print(f"RGB base model: {args.rgb_model_path}")
    print(f"Wavelet base model: {args.wavelet_model_path}")
    
    # Create options object
    opt = EvaluationOptions(
        dataroot=args.dataroot,
        checkpoint_path=checkpoint_path,
        rgb_model_path=args.rgb_model_path,
        wavelet_model_path=args.wavelet_model_path,
        checkpoint_dir=checkpoint_dir,
        name=args.name,
        epoch=epoch
    )
    opt.batch_size = int(args.batch_size)
    opt.gpu_ids = [args.gpu_id] if args.gpu_id >= 0 and torch.cuda.is_available() else []
    
    # Temporarily set isTrain=True to prevent automatic checkpoint loading
    opt.isTrain = True
    
    # Load model (this will load RGB and Wavelet base models too)
    print("Loading MHA fusion model with base models...")
    model = MHAFusionTrainer(opt)
    device = model.device
    
    # Set back to evaluation mode
    opt.isTrain = False
    
    # Load MHA fusion checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading MHA fusion weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if checkpoint is dict with 'model' key or direct state_dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Loaded model from checkpoint dictionary")
        else:
            state_dict = checkpoint
            print("Loaded model state_dict directly")
        
        model.model.load_state_dict(state_dict)
        print("MHA fusion checkpoint loaded successfully!")
    else:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    model.eval()
    print(f"Model loaded successfully on device: {device}")
    
    # Create dual-input dataloader (RGB + Wavelet)
    print(f"Loading validation data from: {args.dataroot}")
    from data import create_mha_dataloader
    data_loader = create_mha_dataloader(opt)
    print(f"Total batches: {len(data_loader)}")
    
    # Run evaluation
    y_true, y_scores = run_evaluation(model, data_loader, device)
    
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
    print(f"  - confusion_matrix.png: Confusion matrix")
    print(f"  - far_frr_curves.png: FAR and FRR curves with EER")
    print(f"  - metrics_summary.png: Bar chart of all metrics")
    print(f"  - score_distribution.png: Distribution of prediction scores")
    print(f"  - per_class_metrics.png: Per-class performance comparison")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
