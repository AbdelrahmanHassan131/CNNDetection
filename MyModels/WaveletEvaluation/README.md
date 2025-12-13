# WaveletsPacketsScratch_128 Model Evaluation - Quick Start Guide

## Overview

This evaluation script calculates comprehensive metrics and generates publication-quality visualizations for the Wavelet Packet-based deepfake detection model (Wolter et al. 2022 architecture).

## Model Details

- **Architecture**: Custom CNN trained from scratch (no ImageNet pretraining)
- **Input**: Wavelet packet coefficients (192 channels = 3 RGB × 64 packets)
- **Wavelet Parameters**:
  - Type: Haar wavelet
  - Level: 3 (generates 4^3 = 64 packets per channel)
  - Mode: Reflect padding
  - Log-scaling: Applied to packet coefficients
- **Input Size**: 128×128 pixels

## Installation

```bash
# Install required packages (if not already installed)
pip install pywt matplotlib seaborn tqdm scikit-learn
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
    --checkpoint "G:\checkpoints\WaveletModel\model_epoch_best.pth" \
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

- **Wavelet Computation**: The dataloader automatically computes wavelet packet coefficients from RGB images during loading
- **Preprocessing**: Images are resized to 128×128 pixels
- **GPU Usage**: Automatically uses GPU if available
- **Batch Processing**: Wavelets are computed in the dataloader for efficiency
- **No Multi-processing**: `num_workers=0` to avoid Windows pickle errors with wavelet transforms

## Model Architecture Details

The WaveletPacketCNN architecture consists of:

1. **Input**: 192 channels (wavelet packets)
2. **Conv Blocks**: 4 convolutional blocks (64→128→256→512 channels)
3. **Pooling**: Max pooling after each block
4. **Global Pooling**: Adaptive average pooling
5. **Classifier**: 
   - Linear(512 → 128) + ReLU + Dropout
   - Linear(128 → 1) for binary classification

## Troubleshooting

### "FileNotFoundError: Checkpoint not found"
- Verify the checkpoint path is correct
- Ensure the `.pth` file exists at the specified location

### "CUDA out of memory"
- Reduce batch size: `--batch_size 16`

### "Dataloader errors"
- Ensure your validation folder has `fake` and `real` subfolders
- Check that images are in supported formats (jpg, png)

## Citation

If using this model architecture, please cite:

> Wolter, Schae

ffer, and Reinhardt. "Wavelet-Packets for Deepfake Image Analysis and Detection."  
> Machine Learning, ECML PKDD 2022 Journal Track.
