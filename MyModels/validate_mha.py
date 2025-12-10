import torch
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from data import create_mha_dataloader


def validate_mha(model, opt):
    """
    Validation function for MHA Fusion model.

    Args:
        model: MHAFusionTrainer instance
        opt: validation options

    Returns:
        tuple: (acc, ap, r_acc, f_acc, y_true, y_pred)
    """
    # Create dataloader with dual inputs
    data_loader = create_mha_dataloader(opt)

    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for rgb_img, wavelet_img, label in data_loader:
            # Move to device
            rgb_img = rgb_img.cuda(non_blocking=True)
            wavelet_img = wavelet_img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # Forward pass through fusion model
            # model.__call__(rgb_input, wavelet_input) handles the dual inputs
            preds = model(rgb_img, wavelet_img).sigmoid().flatten()

            # Collect predictions
            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(label.flatten().cpu().numpy().tolist())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    from options.test_options import TestOptions
    from networks.MHA_128.Trainer_MHA_128 import MHAFusionTrainer

    opt = TestOptions().parse(print_options=False)

    # Enable wavelet computation for validation
    opt.compute_wavelets = True

    # Create fusion model
    model = MHAFusionTrainer(opt)
    model.cuda()
    model.eval()

    # Validate
    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate_mha(model, opt)

    print("=" * 80)
    print("MHA FUSION MODEL VALIDATION RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Real Images Accuracy: {r_acc:.4f}")
    print(f"Fake Images Accuracy: {f_acc:.4f}")
    print("=" * 80)
