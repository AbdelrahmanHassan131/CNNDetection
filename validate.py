import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader
import pywt


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

# ==========================================================
# 🔹 WAVELET VALIDATION — for models that take 12-channel input
# ==========================================================


def rgb_to_wavelet_tensor(img_tensor):
    """
    Converts a [B, 3, H, W] tensor into a [B, 12, H/2, W/2] wavelet tensor
    using 2D Haar wavelet decomposition.
    """
    B, C, H, W = img_tensor.shape
    wavelet_channels = []

    for c in range(C):
        # pywt requires CPU tensors, so move one channel to CPU
        channel = img_tensor[:, c, :, :].cpu().numpy()
        coeffs = [pywt.dwt2(channel[i], 'haar') for i in range(B)]
        LL = torch.tensor(np.stack([c[0] for c in coeffs])).to(
            img_tensor.device)
        LH = torch.tensor(np.stack([c[1][0]
                          for c in coeffs])).to(img_tensor.device)
        HL = torch.tensor(np.stack([c[1][1]
                          for c in coeffs])).to(img_tensor.device)
        HH = torch.tensor(np.stack([c[1][2]
                          for c in coeffs])).to(img_tensor.device)
        wavelet_channels.extend([LL, LH, HL, HH])

    # Stack to form 12 channels
    wavelet_tensor = torch.stack(wavelet_channels, dim=1)  # [B, 12, H/2, W/2]
    return wavelet_tensor


def validate_wavelet(model, opt):
    """
    Validation for wavelet-based models.
    """
    data_loader = create_dataloader(opt)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for img, label in data_loader:
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # ✅ Convert RGB → Wavelet (12 channels)
            wavelet_img = rgb_to_wavelet_tensor(img)

            # Pass through model
            preds = model(wavelet_img).sigmoid().flatten()

            # Move to CPU for numpy
            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(label.flatten().cpu().numpy().tolist())

    # Convert to numpy for metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return acc, ap, r_acc, f_acc, y_true, y_pred


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
