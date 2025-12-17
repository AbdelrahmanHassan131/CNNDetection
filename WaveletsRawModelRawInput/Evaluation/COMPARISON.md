# Evaluation Setup Comparison

## Overview

This document compares the evaluation setups for two wavelet-based deepfake detection models:

1. **WaveletsPacketsScratch_128** (in `CNNDetection/MyModels/WaveletEvaluation/`)
2. **WaveletsRawModelRawInput** (in `CNNDetection/WaveletsRawModelRawInput/Evaluation/`)

Both models use the same wavelet packet architecture from Wolter et al. 2022, but with different training approaches.

---

## Model Comparison

| Aspect | WaveletsPacketsScratch_128 | WaveletsRawModelRawInput |
|--------|---------------------------|--------------------------|
| **Location** | `CNNDetection/MyModels/WaveletEvaluation/` | `CNNDetection/WaveletsRawModelRawInput/Evaluation/` |
| **Trainer** | `Trainer_WaveletsPacketsScratch_128.py` | `trainer_WaveletsRawModelRawInput.py` |
| **Data Loader** | Uses `data.create_dataloader()` | Uses `data_wavelets.create_dataloader_wavelet()` |
| **Training Approach** | May include augmentation | Clean training without augmentation |
| **Validation** | Standard validation | Custom `validate_wavelet()` function |
| **Model Class** | `WolterWaveletPacketTrainer` | `WaveletPacketTrainer` |

---

## Architecture Similarities

Both models share:
- **Wavelet Type**: Haar
- **Decomposition Level**: 3
- **Input Channels**: 192 (3 RGB × 64 wavelet packets)
- **CNN Architecture**: 64→128→256→512 progressive layers
- **Image Size**: Resize to 256, crop to 224
- **Output**: Single logit for binary classification

---

## Evaluation Script Differences

### Import Statements

**WaveletsPacketsScratch_128:**
```python
from networks.FrequencyModels.WaveletsPacketsScratch_128.Trainer_WaveletsPacketsScratch_128 import WolterWaveletPacketTrainer
from visualizations import create_all_visualizations
```

**WaveletsRawModelRawInput:**
```python
from trainer_WaveletsRawModelRawInput import WaveletPacketTrainer
from visualizations import create_all_visualizations
```

### Data Loader Creation

**WaveletsPacketsScratch_128:**
```python
from data import create_dataloader
data_loader = create_dataloader(opt)
```

**WaveletsRawModelRawInput:**
```python
from data_wavelets import create_dataloader_wavelet
data_loader = create_dataloader_wavelet(opt)
```

### Options Configuration

Both use similar options, but with different defaults:

**Common Options:**
- `loadSize = 256`
- `cropSize = 224`
- `wavelet_type = 'haar'`
- `wavelet_level = 3`
- `use_log_packets = True`
- `batch_size = 32`

**WaveletsPacketsScratch_128 Specific:**
```python
opt.compute_wavelets = True  # Tell dataloader to compute wavelets
```

**WaveletsRawModelRawInput Specific:**
```python
# Uses dedicated wavelet dataloader, no special flag needed
```

---

## Output Structure

Both evaluation setups produce identical output files:

### Metrics File
- `metrics.json` - Complete evaluation metrics in JSON format

### Visualization Files
- `roc_curve.png` - ROC curve with AUC and EER
- `precision_recall_curve.png` - Precision-Recall curve
- `confusion_matrix.png` - Confusion matrix (counts and normalized)
- `far_frr_curves.png` - FAR and FRR curves
- `metrics_summary.png` - Bar chart of metrics
- `score_distribution.png` - Score distribution by class
- `per_class_metrics.png` - Per-class performance

---

## Usage Examples

### WaveletsPacketsScratch_128

```bash
cd CNNDetection/MyModels/WaveletEvaluation/

python evaluate_model.py \
    --dataroot "G:/datasets/validation" \
    --checkpoint "path/to/checkpoint.pth" \
    --output_dir "./results/wavelet_packets_eval"
```

### WaveletsRawModelRawInput

```bash
cd CNNDetection/WaveletsRawModelRawInput/Evaluation/

python evaluate_model.py \
    --dataroot "G:/datasets/validation" \
    --checkpoint "../checkpoints/wavelet_raw_experiment/latest_net.pth" \
    --output_dir "./results/wavelet_raw_eval"
```

---

## Key Takeaways

1. **Same Core Architecture**: Both models use identical wavelet packet CNN architecture
2. **Different Training Pipelines**: Different approaches to data loading and training
3. **Identical Evaluation Metrics**: Both produce the same comprehensive evaluation outputs
4. **Separate Evaluation Directories**: Each model has its own evaluation setup
5. **Consistent Visualization**: Both use the same visualization module for publication-quality plots

---

## When to Use Which Model

### Use WaveletsPacketsScratch_128 if:
- You trained with the standard CNNDetection pipeline
- You used the `Trainer_WaveletsPacketsScratch_128.py` trainer
- Your checkpoint is in the standard format

### Use WaveletsRawModelRawInput if:
- You trained with the clean wavelet pipeline
- You used the `trainer_WaveletsRawModelRawInput.py` trainer
- You want to evaluate without augmentation effects
- Your training used `data_wavelets.py` dataloader

---

## Metrics Produced (Both Models)

### Basic Metrics
- Accuracy, Precision, Recall, F1-Score

### Advanced Metrics
- ROC-AUC, Average Precision (AP)
- Equal Error Rate (EER)
- Specificity, Sensitivity

### Error Rates
- False Accept Rate (FAR)
- False Reject Rate (FRR)

### Confusion Matrix
- True Positives, True Negatives
- False Positives, False Negatives

### Per-Class Metrics
- Precision, Recall, F1 for Real and Fake classes
- AUC for each class

---

## Directory Structure

```
CNNDetection/
├── MyModels/
│   └── WaveletEvaluation/
│       ├── evaluate_model.py
│       ├── visualizations.py
│       └── README.md
│
└── WaveletsRawModelRawInput/
    ├── Evaluation/
    │   ├── evaluate_model.py
    │   ├── visualizations.py
    │   ├── README.md
    │   └── run_evaluation_examples.py
    ├── trainer_WaveletsRawModelRawInput.py
    ├── data_wavelets.py
    ├── validate_wavelet.py
    └── train_WaveletsRawModelRawInput.py
```

---

## Conclusion

Both evaluation setups provide comprehensive, publication-quality evaluation for their respective models. The main difference lies in the model loading and data preprocessing, while the evaluation metrics and visualizations remain consistent across both implementations.
