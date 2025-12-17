# WaveletsRawModelRawInput Model Evaluation

This directory contains comprehensive evaluation tools for the **WaveletsRawModelRawInput** deepfake detection model.

## Model Architecture

The WaveletsRawModelRawInput model is based on the Wolter et al. 2022 architecture:
- **Input**: Wavelet packet coefficients (192 channels for level-3 Haar wavelets on RGB images)
- **Architecture**: Custom CNN with progressive layers (64→128→256→512)
- **Training**: Clean training without augmentation
- **Image Processing**: Resize to 256, center crop to 224

## Files

- **`evaluate_model.py`**: Main evaluation script
- **`visualizations.py`**: Visualization generation module
- **`README.md`**: This file

## Usage

### Basic Evaluation

```bash
python evaluate_model.py \
    --dataroot "path/to/validation/dataset" \
    --checkpoint "path/to/checkpoint.pth" \
    --output_dir "./evaluation_results"
```

### Full Example

```bash
python evaluate_model.py \
    --dataroot "G:/datasets/validation" \
    --checkpoint "../checkpoints/wavelet_raw_experiment/latest_net.pth" \
    --name "wavelet_raw_experiment" \
    --output_dir "./results/wavelet_raw_eval" \
    --batch_size 32 \
    --threshold 0.5 \
    --gpu_id 0
```

### Arguments

- `--dataroot`: Path to validation dataset (must contain `fake/` and `real/` subdirectories)
- `--checkpoint`: Path to model checkpoint file (`.pth`)
- `--name`: Experiment name (default: `wavelet_raw_experiment`)
- `--checkpoints_dir`: Directory containing checkpoints (default: `./checkpoints`)
- `--output_dir`: Directory to save evaluation results (default: `./evaluation_results`)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--threshold`: Decision threshold for binary classification (default: 0.5)
- `--gpu_id`: GPU ID to use, -1 for CPU (default: 0)

## Output Files

The evaluation script generates the following files in the output directory:

### Metrics
- **`metrics.json`**: Complete metrics in JSON format including:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Average Precision (AP)
  - Equal Error Rate (EER)
  - Confusion Matrix values
  - Specificity, Sensitivity
  - False Accept Rate (FAR), False Reject Rate (FRR)
  - Per-class metrics (Real/Fake)

### Visualizations (Publication-Quality PNG)
- **`roc_curve.png`**: ROC curve with AUC and EER marked
- **`precision_recall_curve.png`**: Precision-Recall curve with AP
- **`confusion_matrix.png`**: Confusion matrix (counts and normalized)
- **`far_frr_curves.png`**: FAR and FRR curves with EER intersection
- **`metrics_summary.png`**: Bar chart of all main metrics
- **`score_distribution.png`**: Distribution of prediction scores by class
- **`per_class_metrics.png`**: Per-class performance comparison

## Dataset Structure

Your validation dataset should be organized as:
```
validation_dataset/
├── fake/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
└── real/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

## Model Details

### Wavelet Parameters
- **Wavelet Type**: Haar
- **Decomposition Level**: 3
- **Signal Extension Mode**: Reflect
- **Log Scaling**: Enabled

### Data Processing
- **Load Size**: 256 (resize)
- **Crop Size**: 224 (center crop)
- **Input Channels**: 192 (3 RGB channels × 64 wavelet packets)
- **Output**: Single logit (binary classification)

### Key Differences from WaveletsPacketsScratch_128
- Uses the same wavelet packet architecture
- Clean training without augmentation
- Direct wavelet packet computation in dataloader
- Simplified training pipeline

## Metrics Explanation

### Basic Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of predicted fakes, how many are actually fake
- **Recall**: Of actual fakes, how many are detected
- **F1-Score**: Harmonic mean of precision and recall

### Advanced Metrics
- **ROC-AUC**: Area under ROC curve (threshold-independent)
- **AP (Average Precision)**: Area under Precision-Recall curve
- **EER (Equal Error Rate)**: Point where FAR = FRR
- **Specificity**: True negative rate
- **Sensitivity**: True positive rate (same as recall)

### Error Rates
- **FAR (False Accept Rate)**: Fake images incorrectly classified as real
- **FRR (False Reject Rate)**: Real images incorrectly classified as fake

## Example Output

```
==============================================================
EVALUATION METRICS SUMMARY - WaveletsRawModelRawInput
==============================================================

Dataset Information:
  Total Samples: 5000
  Real Images: 2500
  Fake Images: 2500

Basic Classification Metrics:
  Accuracy:    0.9234
  Precision:   0.9156
  Recall:      0.9312
  F1-Score:    0.9233

Advanced Metrics:
  ROC-AUC:     0.9678
  AP (Average Precision): 0.9645
  EER (Equal Error Rate): 0.0823 (at threshold 0.4567)

Confusion Matrix:
  True Negatives:  2301
  False Positives: 199
  False Negatives: 172
  True Positives:  2328

...
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PyWavelets (pywt)

## Notes

1. **GPU Memory**: Evaluation uses less memory than training. Batch size of 32 should work on most GPUs.
2. **Reproducibility**: Set `serial_batches=True` for consistent ordering (already set in evaluation mode).
3. **Threshold Selection**: Default is 0.5, but you can adjust based on your use case. Check EER threshold in results for optimal balanced performance.
4. **Wavelet Computation**: Wavelets are computed on-the-fly by the dataloader, matching the training process.

## Troubleshooting

### Checkpoint Not Found
Ensure the checkpoint path is correct. The script accepts both:
- Direct path: `path/to/checkpoint.pth`
- Epoch name: `latest` (will look in `checkpoints_dir/name/latest_net.pth`)

### CUDA Out of Memory
Reduce batch size: `--batch_size 16` or `--batch_size 8`

### Wrong Input Shape
Ensure you're using the wavelet dataloader from `data_wavelets.py`. The model expects 192-channel wavelet packet input.

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{wolter2022wavelet,
  title={Wavelet-Packets for Deepfake Image Analysis and Detection},
  author={Wolter, Raimund and Blanke, Ulf and Agarwal, Shubham},
  journal={Machine Learning, ECML PKDD 2022 Journal Track},
  year={2022}
}
```

## Contact

For issues or questions about this evaluation setup, please refer to the main project documentation.
