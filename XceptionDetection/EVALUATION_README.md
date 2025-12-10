# Model Evaluation - Quick Start Guide

## Overview

This evaluation script calculates comprehensive metrics and generates publication-quality visualizations for your trained Xception deepfake detection model.

## Installation

```bash
pip install matplotlib seaborn tqdm
```

(Other dependencies should already be installed from training)

## Usage

### Basic Command

```bash
python evaluate_model.py --dataroot <path_to_val_data> --checkpoint <checkpoint_path>
```

### Example

```bash
python evaluate_model.py \
    --dataroot "./dataset/val/" \
    --checkpoint "./checkpoints/xception_experiment/latest_net.pth" \
    --output_dir "./evaluation_results"
```

## Dataset Structure

Your validation dataset should have this structure:

```
validation_data/
├── fake/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── real/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Label Mapping:**
- `fake` folder → label 0
- `real` folder → label 1

## Output Files

The script generates:

### Metrics File
- **metrics.json**: All calculated metrics in JSON format

### Visualizations (300 DPI, publication-quality)
- **roc_curve.png**: ROC curve with AUC and EER marked
- **precision_recall_curve.png**: PR curve with AP score
- **confusion_matrix.png**: Confusion matrix (counts + normalized)
- **far_frr_curves.png**: FAR/FRR curves with EER intersection
- **metrics_summary.png**: Bar chart of all metrics
- **score_distribution.png**: Score distributions by class
- **per_class_metrics.png**: Per-class performance comparison

## Calculated Metrics

### Basic Metrics
- Accuracy, Precision, Recall, F1-Score

### Advanced Metrics
- **ROC-AUC**: Area under ROC curve
- **AP**: Average Precision
- **EER**: Equal Error Rate (common in deepfake research)

### Diagnostic Metrics
- Specificity, Sensitivity
- **FAR**: False Accept Rate
- **FRR**: False Reject Rate

### Per-Class Metrics
- Precision, Recall, F1 for Real and Fake classes
- **AUC per class**: Separate AUC scores

## Command-Line Arguments

- `--dataroot`: Path to validation dataset (required)
- `--checkpoint`: Path to checkpoint file (required)
- `--output_dir`: Output directory (default: `./evaluation_results`)
- `--batch_size`: Batch size (default: 32)
- `--threshold`: Decision threshold (default: 0.5)
- `--gpu_id`: GPU ID or -1 for CPU (default: 0)

## Example Output

```
==============================================================
EVALUATION METRICS SUMMARY
==============================================================

Dataset Information:
  Total Samples: 10000
  Real Images: 5000
  Fake Images: 5000

Basic Classification Metrics:
  Accuracy:    0.9542
  Precision:   0.9523
  Recall:      0.9561
  F1-Score:    0.9542

Advanced Metrics:
  ROC-AUC:     0.9876
  AP (Average Precision): 0.9854
  EER (Equal Error Rate): 0.0234

...
```

## Notes

- All images are automatically preprocessed (resize 333, crop 299, normalize 0.5)
- GPU is used automatically if available
- All visualizations are 300 DPI for publication quality
