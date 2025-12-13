# MHA_128 Fusion Model Evaluation - Quick Start Guide

## Overview

This evaluation script calculates comprehensive metrics for the MHA_128 model - a Multi-Head Attention fusion model that combines RGB (wang2020_128) and Wavelet (WaveletsPacketsScratch_128) features for enhanced deepfake detection.

## Model Details

- **Architecture**: Multi-Head Attention Fusion
- **Base Models**:
  - **RGB Branch**: wang2020_128 (ResNet50 with custom FC)
  - **Wavelet Branch**: WaveletsPacketsScratch_128 (Custom CNN on wavelet packets)
- **Fusion Method**: Cross-attention between RGB and Wavelet embeddings
- **Frozen Base**: Pre-trained base models are frozen, only fusion layers are trained
- **Input Size**: 224×224 (after resizing from 256)
- **Dual Input**: Requires both RGB images and wavelet packet coefficients

## Prerequisites

You **must** have trained checkpoints for:
1. **wang2020_128** - RGB model
2. **WaveletsPacketsScratch_128** - Wavelet model
3. **MHA_128** - Fusion model

## Installation

```bash
# Required packages should already be installed
# pywt for wavelet transforms is essential
```

## Usage

### Basic Command

```bash
python evaluate_model.py \
    --dataroot <path_to_val_data> \
    --checkpoint <mha_fusion_checkpoint> \
    --rgb_model_path <wang2020_checkpoint> \
    --wavelet_model_path <wavelet_checkpoint>
```

### Example

```bash
python evaluate_model.py \
    -- dataroot "G:\path\to\dataset\val" \
    --checkpoint "G:\checkpoints\MHA_128\model_epoch_best.pth" \
    --rgb_model_path "G:\checkpoints\wang2020_128\model_epoch_best.pth" \
    --wavelet_model_path "G:\checkpoints\WaveletsPacketsScratch_128\model_epoch_best.pth" \
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

The script generates the same comprehensive outputs as other models:

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

## Command-Line Arguments

### Required:
- `--dataroot`: Path to validation dataset
- `--checkpoint`: Path to MHA fusion checkpoint file
- `--rgb_model_path`: Path to wang2020_128 checkpoint (**required**)
- `--wavelet_model_path`: Path to WaveletsPacketsScratch_128 checkpoint (**required**)

### Optional:
- `--output_dir`: Output directory (default: `./evaluation_results`)
- `--batch_size`: Batch size (default: 32)
- `--threshold`: Decision threshold (default: 0.5)
- `--gpu_id`: GPU ID or -1 for CPU (default: 0)

## Model Architecture Details

The MHA_128 fusion model:

1. **RGB Processing**:
   - Images → ResNet50 → 128-dim embeddings
   - Base model frozen during fusion training

2. **Wavelet Processing**:
   - Images → Wavelet Packet Transform → Custom CNN → 128-dim embeddings
   - Base model frozen during fusion training

3. **Fusion Layer**:
   - Cross-Attention: RGB attends to Wavelet AND Wavelet attends to RGB
   - Multihead Attention (4 heads, 128-dim)
   - Feed-Forward Networks
   - Final Classifier: 128 → 64 → 1

4. **Training Strategy**:
   - Base models frozen (no gradient updates)
   - Only fusion and classifier layers trained
   - Allows leveraging pre-trained features without catastrophic forgetting

## Important Notes

- **Dual Dataloader**: Uses `create_mha_dataloader` which returns `(rgb, wavelet, label)` tuples
- **Base Models Required**: MHA cannot run without valid wang2020_128 and Wavelet checkpoints
- **Preprocessing**: Same as individual models (resize 256, crop 224)
- **GPU Recommended**: Processing both RGB and wavelets is compute-intensive

## Expected Performance

The fusion model should achieve **equal or better** performance than either base model alone, as it combines complementary information:
- **RGB**: Spatial and color patterns
- **Wavelet**: Frequency-domain artifacts

Typical gains: **1-3% improvement in AUC** over best individual model.

## Troubleshooting

### "Error: RGB model not found"
- Verify `--rgb_model_path` points to a valid wang2020_128 checkpoint
- Ensure the file exists and is readable

### "Error: Wavelet model not found"
- Verify `--wavelet_model_path` points to a valid WaveletsPacketsScratch_128 checkpoint
- Ensure the file exists

### "Architecture mismatch" errors
- Ensure RGB checkpoint is from wang2020_128 (not WangRawModel)
- Ensure Wavelet checkpoint has matching wavelet parameters (haar, level 3)

### "Dataloader errors"
- The MHA dataloader automatically computes wavelets
- Ensure validation folder has `fake` and `real` subfolders
- Check memory if batches are too large

## Citation

If using this fusion approach, consider citing the original Multi-Head Attention mechanisms and deepfake detection works that inspired this architecture.
