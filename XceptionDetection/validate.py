import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score

from data import create_dataloader


def validate(model, opt):
    """Validate model and return accuracy and average precision"""
    data_loader = create_dataloader(opt)
    
    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            img, label = data
            if opt.gpu_ids:
                img = img.cuda()
            
            output = model(img)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            
            y_pred.extend(pred.flatten().tolist() if pred.ndim > 0 else [float(pred)])
            y_true.extend(label.numpy().flatten().tolist())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    
    return acc, ap
