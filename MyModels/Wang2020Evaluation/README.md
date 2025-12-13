# wang2020_128 Model Evaluation - Quick Start Guide

## Overview

This evaluation script calculates comprehensive metrics and generates publication-quality visualizations for the wang2020_128 deepfake detection model (ResNet50 with custom FC layer).

## Model Details

- **Architecture**: ResNet50 pretrained on ImageNet
- **FC Layer**: Custom sequential layer
  - Linear(2048 → 128)
  - ReLU activation
  - Dropout(0.5)
  - Linear(128 → 1) for binary classification
- **Input**: RGB images (3 channels)
- **Input Size**: 128×128 pixels
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Installation

```bash
# Required packages should already be installed
# If not: pip install torch torchvision matplotlib seaborn tqdm scikit-learn
```

## Usage

### Basic Command

```bash
python evaluate_model.py --dataroot <path_to_val_data> --checkpoint <checkpoint_path>
```

### Example

```bash
python evaluate_model.py \
    --dataroot "G:\path\to\dataset\val" \
    --checkpoint "G:\checkpoints\wang2020_128\model_epoch_best.pth" \
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
- **EER**: Equal Error Rate

### Diagnostic Metrics
- Specificity, Sensitivity
- **FAR**: False Accept Rate
- **FRR**: False Reject Rate

### Per-Class Metrics
- Precision, Recall, F1 for Real and Fake classes
- AUC per class

## Command-Line Arguments

- `--dataroot`: Path to validation dataset (required)
- `--checkpoint`: Path to checkpoint file (required)
- `--output_dir`: Output directory (default: `./evaluation_results`)
- `--batch_size`: Batch size (default: 32)
- `--threshold`: Decision threshold (default: 0.5)
- `--gpu_id`: GPU ID or -1 for CPU (default: 0)

## Important Notes

- **Preprocessing**: Images are resized to 128×128 pixels and normalized with ImageNet statistics
- **GPU Usage**: Automatically uses GPU if available
- **Architecture**: This model has an intermediate 128-dim embedding layer (unlike standard ResNet50)
- **No Multi-processing**: `num_workers=0` to avoid Windows pickle errors

## Model Architecture Details

The wang2020_128 architecture consists of:

1. **Backbone**: ResNet50 pretrained on ImageNet
2. **Modified FC Layer**:
   - Linear(2048 → 128) - Creates dense embeddings
   - ReLU activation
   - Dropout(p=0.5) - Regularization
   - Linear(128 → 1) - Binary classification output
3. **Loss**: BCEWithLogitsLoss (Binary Cross-Entropy with Logits)

## Comparison with WangRawModel

**Similarities:**
- Both use ResNet50 backbone
- Both use ImageNet pretraining
- Both use BCE loss

**Differences:**
- **wang2020_128**: Has intermediate 128-dim layer: `2048→128→1`
- **WangRawModel**: Direct classification: `2048→1`  
- **wang2020_128**: Uses 128×128 input
- **WangRawModel**: Uses 224×224 input

## Troubleshooting

### "FileNotFoundError: Checkpoint not found"
- Verify the checkpoint path is correct
- Ensure the `.pth` file exists

### "CUDA out of memory"
- Reduce batch size: `--batch_size 16`

### "Dataloader errors"
- Ensure validation folder has `fake` and `real` subfolders
- Check images are in supported formats (jpg, png)

## Expected Performance

This model should achieve high accuracy (>95%) on standard deepfake datasets if properly trained, as it uses a proven ResNet50 backbone with additional embedding capacity.
