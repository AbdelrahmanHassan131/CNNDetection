"""
Custom validation function for Wavelet Packet model.
Uses wavelet dataloader instead of standard RGB dataloader.
"""

import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from WaveletsRawModelRawInput.data_wavelets import create_dataloader_wavelet


def validate_wavelet(model, opt):
    """
    Validate wavelet model using wavelet packet dataloader.
    
    Args:
        model: The wavelet model to validate
        opt: Options object with validation settings
    
    Returns:
        tuple: (accuracy, average_precision, real_accuracy, fake_accuracy, y_true, y_pred)
    """
    # Create wavelet dataloader for validation
    data_loader = create_dataloader_wavelet(opt)
    
    with torch.no_grad():
        y_true, y_pred = [], []
        for wavelet_packets, label in data_loader:
            # Move to GPU
            in_tens = wavelet_packets.cuda()
            
            # Forward pass - model expects wavelet packets
            output = model(in_tens).sigmoid().flatten().tolist()
            y_pred.extend(output)
            y_true.extend(label.flatten().tolist())
    
    # Convert to numpy arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate metrics
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    
    return acc, ap, r_acc, f_acc, y_true, y_pred
