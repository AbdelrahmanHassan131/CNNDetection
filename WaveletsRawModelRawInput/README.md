# WaveletsRawModelRawInput - Original Wolter et al. 2022

## Overview

This is a clean implementation of the Wavelet Packet-based deepfake detection model from:

> **Wolter, Schaeffer, and Reinhardt.** "Wavelet-Packets for Deepfake Image Analysis and Detection."  
> Machine Learning, ECML PKDD 2022 Journal Track.

## Key Differences from Modified Version

### Original Architecture (This Implementation):
- **Classifier**: `512 → Dropout → 1` (direct classification)
- **No intermediate embedding layer**
- **Exactly as described in the paper**

### Modified Version (WaveletsPacketsScratch_128):
- **Classifier**: `512 → 128 → ReLU → Dropout → 1`
- **Has 128-dim embedding layer**
- **Custom modification**

## Model Details

- **Input**: Wavelet packet coefficients (192 channels = 3 RGB × 64 packets)
- **Wavelet Parameters**:
  - Type: Haar wavelet
  - Level: 3 (generates 4^3 = 64 packets per channel)
  - Mode: Reflect padding
  - Log-scaling: Applied to packet coefficients
- **Input Size**: 224×224 pixels
- **Architecture**: Custom CNN trained from scratch (no ImageNet pretraining)

## Clean Data Processing

This implementation uses **NO augmentation**:
- ✅ Resize to 256 → Center crop to 224
- ✅ Wavelet packet computation
- ✅ Log-scaling of coefficients
- ❌ No blur
- ❌ No JPEG compression
- ❌ No random flipping
- ❌ No color jittering

## Files

- `trainer_WaveletsRawModelRawInput.py` - Model trainer (original architecture)
- `data_wavelets.py` - Clean dataloader (no augmentation)
- `train_WaveletsRawModelRawInput.py` - Training script
- `__init__.py` - Module initialization

## Usage

### Training

```bash
python WaveletsRawModelRawInput/train_WaveletsRawModelRawInput.py \
    --dataroot "G:\path\to\dataset" \
    --name WaveletOriginal \
    --train_split train \
    --val_split val \
    --batch_size 32 \
    --lr 0.001 \
    --niter 100 \
    --num_threads 0
```

### Testing Dataloader

```bash
python WaveletsRawModelRawInput/data_wavelets.py \
    --dataroot "G:\path\to\val" \
    --batch_size 32 \
    --image_size 224
```

## Dataset Structure

```
dataset/
├── train/
│   ├── fake/
│   │   ├── image1.jpg
│   │   └── ...
│   └── real/
│       ├── image1.jpg
│       └── ...
└── val/
    ├── fake/
    └── real/
```

## Architecture Details

### Convolutional Blocks:
1. **Block 1**: 192 → 64 channels (Conv + BN + ReLU + MaxPool)
2. **Block 2**: 64 → 128 channels (Conv + BN + ReLU + MaxPool)
3. **Block 3**: 128 → 256 channels (Conv + BN + ReLU + MaxPool)
4. **Block 4**: 256 → 512 channels (Conv + BN + ReLU + MaxPool)

### Classifier:
- Global Average Pooling: 512 → 512×1×1
- Dropout (p=0.5)
- Linear: 512 → 1 (binary classification)

### Loss:
- BCEWithLogitsLoss (Binary Cross-Entropy with Logits)

### Optimizer:
- Adam (default) or SGD
- Weight decay: 1e-4
- Learning rate: 0.001 (default)

## Wavelet Packet Computation

The model uses **Haar wavelets** with **level-3 decomposition**:

1. **Input**: RGB image (H, W, 3)
2. **Per-channel decomposition**: Each RGB channel → 64 wavelet packets
3. **Total channels**: 3 × 64 = 192
4. **Output size**: After 4 pooling layers, (H/16, W/16)
   - For 224×224 input → 192 × 14 × 14

## Expected Performance

When trained properly, this model should achieve:
- **Accuracy**: >95% on standard deepfake datasets
- **ROC-AUC**: >98%
- **Comparable to ResNet-based models** while using frequency-domain features

## Citation

If using this implementation, please cite:

```bibtex
@article{wolter2022wavelet,
  title={Wavelet-Packets for Deepfake Image Analysis and Detection},
  author={Wolter, Moritz and Schaeffer, Felix and Reinhardt, Jochen},
  journal={Machine Learning},
  year={2022},
  publisher={Springer}
}
```

## Notes

- This is the **original paper architecture** without modifications
- For the version with 128-dim embeddings, see `MyModels/networks/FrequencyModels/WaveletsPacketsScratch_128/`
- Wavelet computation happens in the dataloader (CPU-parallel) for efficiency
- Use `num_threads=0` on Windows to avoid multiprocessing errors
