"""
Grad-CAM Visualization for Deepfake Detection Models
Compares feature extraction across RGB, Wavelet, and MHA Fusion models.

Usage:
    python gradcam_visualization.py \
        --rgb_model_path /path/to/rgb_model.pth \
        --wavelet_model_path /path/to/wavelet_model.pth \
        --checkpoint_path /path/to/fusion_model.pth \
        --data_root /path/to/images \
        --output_dir ./gradcam_results \
        --num_images 8
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
import cv2
from collections import OrderedDict

from options.tSNE_options import tSNE_Options

# ============================================================================
# Grad-CAM Implementation
# ============================================================================


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(
            self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input to the model
            target_class: Target class index (None for binary classification)

        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            # For binary classification, use the output directly
            target = output.squeeze()
        else:
            target = output[0, target_class]

        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(
            2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1,
                        keepdim=True)  # (B, 1, H, W)

        # ReLU to keep only positive influences
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, output.detach().cpu()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
    """

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            target = output.squeeze()
        else:
            target = output[0, target_class]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Grad-CAM++ weighting
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        weights = torch.sum(alpha * F.relu(gradients),
                            dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, output.detach().cpu()


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_rgb_model(model_path, device):
    """Load pre-trained RGB model (Wang2020 ResNet50)"""
    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location=device)

    # Handle Sequential fc structure
    if 'fc.0.weight' in state_dict.get('model', state_dict):
        in_features = 2048
        model.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    model_state = state_dict.get('model', state_dict)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model


def load_wavelet_model(model_path, device, wavelet_level=3):
    """Load pre-trained Wavelet model (Wolter2022)"""
    from networks.FrequencyModels.WaveletsPacketsScratch_128.Trainer_WaveletsPacketsScratch_128 import (
        WaveletPacketCNN
    )

    num_packets_per_channel = 4 ** wavelet_level
    input_channels = 3 * num_packets_per_channel

    model = WaveletPacketCNN(input_channels=input_channels, num_classes=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    return model


def load_fusion_model(rgb_model_path, wavelet_model_path, checkpoint_path, device, opt=None):
    """Load MHA Fusion model with its base models"""
    # Load RGB model for embedding extraction
    rgb_model = load_rgb_model(rgb_model_path, device)

    # Modify RGB model to output embeddings (128-dim)
    rgb_model.fc = nn.Sequential(
        rgb_model.fc[0],  # Linear(2048 -> 128)
        rgb_model.fc[1],  # ReLU
    )

    # Load Wavelet model for embedding extraction
    wavelet_model = load_wavelet_model(wavelet_model_path, device)
    wavelet_model.classifier = nn.Sequential(
        wavelet_model.classifier[0],  # Linear(512, 128)
        wavelet_model.classifier[1],  # ReLU
    )

    # Load fusion model
    from networks.MHA_128.Trainer_MHA_128 import MHAFusionClassifier

    fusion_model = MHAFusionClassifier(
        embed_dim=128,
        num_heads=4,
        dropout=0.1,
        fusion_type='cross_attention'
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    fusion_model.load_state_dict(state_dict['model'])
    fusion_model.to(device)
    fusion_model.eval()

    return rgb_model, wavelet_model, fusion_model


# ============================================================================
# Wavelet Packet Computation
# ============================================================================

def compute_wavelet_packets(img_tensor, wavelet='haar', level=3, mode='reflect'):
    """
    Compute wavelet packet coefficients for visualization.
    """
    if torch.is_tensor(img_tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor

    # Ensure (H, W, 3) format
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    all_packets = []

    def get_paths(level):
        if level == 0:
            return ['']
        paths = []
        prev_paths = get_paths(level - 1)
        for path in prev_paths:
            for letter in ['a', 'h', 'v', 'd']:
                paths.append(path + letter)
        return paths

    packet_paths = get_paths(level)

    for c in range(3):
        channel = img[:, :, c]
        wp = pywt.WaveletPacket2D(
            data=channel, wavelet=wavelet, mode=mode, maxlevel=level)

        for path in packet_paths:
            coeff = wp[path].data
            all_packets.append(coeff)

    all_packets = np.array(all_packets)
    packets_tensor = torch.tensor(all_packets, dtype=torch.float32)

    # Apply log scaling
    packets_tensor = torch.sign(packets_tensor) * \
        torch.log(torch.abs(packets_tensor) + 1e-10)

    return packets_tensor


# ============================================================================
# Visualization Utilities
# ============================================================================

def overlay_cam_on_image(img, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay Grad-CAM heatmap on original image."""
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = np.float32(heatmap) / 255 + np.float32(img) / 255
    overlay = overlay / overlay.max()

    # Blend
    blended = alpha * (heatmap / 255) + (1 - alpha) * (img / 255)
    blended = np.clip(blended, 0, 1)

    return blended, heatmap / 255


def visualize_wavelet_packets(packets, level=3):
    """
    Create a visualization of wavelet packet coefficients.
    Shows the energy distribution across frequency bands.
    """
    num_packets = 4 ** level

    # Compute energy per packet (summed across channels)
    packets_np = packets.cpu().numpy() if torch.is_tensor(packets) else packets

    # Reshape to (3, num_packets, H, W)
    packets_reshaped = packets_np.reshape(
        3, num_packets, packets_np.shape[-2], packets_np.shape[-1])

    # Sum energy across spatial dimensions
    energy = np.sum(packets_reshaped ** 2, axis=(2, 3))  # (3, num_packets)

    # Average across channels
    avg_energy = np.mean(energy, axis=0)  # (num_packets,)

    # Reshape to 2D grid (8x8 for level 3)
    grid_size = 2 ** level
    energy_grid = avg_energy.reshape(grid_size, grid_size)

    return energy_grid


def create_comparison_figure(images_data, output_path, figsize=(20, 5)):
    """
    Create a comparison figure showing Grad-CAM results for all models.

    Args:
        images_data: List of dicts with keys: 'original', 'rgb_cam', 'wavelet_cam', 
                     'wavelet_vis', 'label', 'rgb_pred', 'wavelet_pred', 'fusion_pred'
    """
    n_images = len(images_data)

    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_images))
    gs = gridspec.GridSpec(n_images, 5, figure=fig, wspace=0.05, hspace=0.2)

    col_titles = ['Original Image', 'RGB Model\nGrad-CAM', 'Wavelet Packets\nEnergy',
                  'Wavelet Model\nGrad-CAM', 'Prediction Summary']

    for i, data in enumerate(images_data):
        # Original image
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(data['original'])
        ax1.axis('off')
        if i == 0:
            ax1.set_title(col_titles[0], fontsize=12, fontweight='bold')
        label_text = 'REAL' if data['label'] == 1 else 'FAKE'
        ax1.text(0.5, -0.1, f'Ground Truth: {label_text}', transform=ax1.transAxes,
                 ha='center', fontsize=10, color='red' if data['label'] == 1 else 'green')

        # RGB Grad-CAM
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(data['rgb_cam'])
        ax2.axis('off')
        if i == 0:
            ax2.set_title(col_titles[1], fontsize=12, fontweight='bold')

        # Wavelet energy visualization
        ax3 = fig.add_subplot(gs[i, 2])
        im = ax3.imshow(data['wavelet_vis'], cmap='hot',
                        interpolation='nearest')
        ax3.axis('off')
        if i == 0:
            ax3.set_title(col_titles[2], fontsize=12, fontweight='bold')

        # Wavelet Grad-CAM
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.imshow(data['wavelet_cam'])
        ax4.axis('off')
        if i == 0:
            ax4.set_title(col_titles[3], fontsize=12, fontweight='bold')

        # Predictions summary
        ax5 = fig.add_subplot(gs[i, 4])
        ax5.axis('off')
        if i == 0:
            ax5.set_title(col_titles[4], fontsize=12, fontweight='bold')

        # Create prediction text
        rgb_prob = torch.sigmoid(data['rgb_pred']).item()
        wav_prob = torch.sigmoid(data['wavelet_pred']).item()
        fus_prob = torch.sigmoid(data['fusion_pred']).item(
        ) if data['fusion_pred'] is not None else None

        pred_text = f"RGB: {rgb_prob:.3f}\n({'REAL' if rgb_prob > 0.5 else 'FAKE'})\n\n"
        pred_text += f"Wavelet: {wav_prob:.3f}\n({'REAL' if wav_prob > 0.5 else 'FAKE'})\n\n"
        if fus_prob is not None:
            pred_text += f"Fusion: {fus_prob:.3f}\n({'REAL' if fus_prob > 0.5 else 'FAKE'})"

        ax5.text(0.5, 0.5, pred_text, transform=ax5.transAxes, ha='center', va='center',
                 fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure to {output_path}")


def create_detailed_cam_figure(img, rgb_cam, wavelet_cam, wavelet_energy,
                               predictions, label, output_path):
    """
    Create a detailed Grad-CAM visualization for a single image.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Original and overlays
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=11)
    axes[0, 0].axis('off')

    # RGB Grad-CAM overlay
    rgb_overlay, rgb_heatmap = overlay_cam_on_image(img, rgb_cam)
    axes[0, 1].imshow(rgb_overlay)
    axes[0, 1].set_title('RGB Model Grad-CAM', fontsize=11)
    axes[0, 1].axis('off')

    # Wavelet Grad-CAM overlay (on wavelet representation)
    axes[0, 2].imshow(wavelet_cam)
    axes[0, 2].set_title('Wavelet Model Grad-CAM', fontsize=11)
    axes[0, 2].axis('off')

    # Wavelet energy
    im = axes[0, 3].imshow(wavelet_energy, cmap='hot')
    axes[0, 3].set_title('Wavelet Packet Energy', fontsize=11)
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 2: Heatmaps and predictions
    axes[1, 0].imshow(rgb_heatmap)
    axes[1, 0].set_title('RGB Heatmap', fontsize=11)
    axes[1, 0].axis('off')

    # RGB CAM raw
    axes[1, 1].imshow(rgb_cam, cmap='jet')
    axes[1, 1].set_title('RGB CAM (Raw)', fontsize=11)
    axes[1, 1].axis('off')

    # Wavelet CAM raw
    axes[1, 2].imshow(predictions.get(
        'wavelet_cam_raw', wavelet_cam), cmap='jet')
    axes[1, 2].set_title('Wavelet CAM (Raw)', fontsize=11)
    axes[1, 2].axis('off')

    # Predictions summary
    axes[1, 3].axis('off')
    label_text = 'REAL' if label == 1 else 'FAKE'

    rgb_prob = predictions['rgb_prob']
    wav_prob = predictions['wavelet_prob']
    fus_prob = predictions.get('fusion_prob', None)

    summary = f"Ground Truth: {label_text}\n\n"
    summary += f"RGB Model:\n  Prob: {rgb_prob:.4f}\n  Pred: {'REAL' if rgb_prob > 0.5 else 'FAKE'}\n\n"
    summary += f"Wavelet Model:\n  Prob: {wav_prob:.4f}\n  Pred: {'REAL' if wav_prob > 0.5 else 'FAKE'}\n\n"
    if fus_prob is not None:
        summary += f"Fusion Model:\n  Prob: {fus_prob:.4f}\n  Pred: {'REAL' if fus_prob > 0.5 else 'FAKE'}"

    axes[1, 3].text(0.1, 0.9, summary, transform=axes[1, 3].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Processing Function
# ============================================================================

def process_images(args):
    """Process images and generate Grad-CAM visualizations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("\nLoading models...")
    rgb_model = load_rgb_model(args.rgb_model_path, device)
    wavelet_model = load_wavelet_model(
        args.wavelet_model_path, device, args.wavelet_level)

    fusion_model = None
    rgb_embed_model = None
    wavelet_embed_model = None

    if args.checkpoint_path:
        rgb_embed_model, wavelet_embed_model, fusion_model = load_fusion_model(
            args.rgb_model_path, args.wavelet_model_path,
            args.checkpoint_path, device
        )

    # Get target layers for Grad-CAM
    # RGB model: last conv layer in layer4
    rgb_target_layer = rgb_model.layer4[-1].conv3

    # Wavelet model: last conv layer (conv4)
    wavelet_target_layer = wavelet_model.conv4

    # Initialize Grad-CAM
    rgb_gradcam = GradCAM(rgb_model, rgb_target_layer)
    wavelet_gradcam = GradCAM(wavelet_model, wavelet_target_layer)

    # Image transforms
    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # Collect image paths
    image_paths = []
    labels = []

    # Check for real/fake subdirectories
    real_dir = os.path.join(args.data_root, '0_fake')
    fake_dir = os.path.join(args.data_root, '1_real')

    if os.path.exists(real_dir) and os.path.exists(fake_dir):
        # Structured directory
        for img_name in sorted(os.listdir(real_dir))[:args.num_images // 2]:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(real_dir, img_name))
                labels.append(0)

        for img_name in sorted(os.listdir(fake_dir))[:args.num_images // 2]:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(fake_dir, img_name))
                labels.append(1)
    else:
        # Flat directory - infer labels from filename
        for img_name in sorted(os.listdir(args.data_root))[:args.num_images]:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.data_root, img_name))
                # Try to infer label from filename
                labels.append(1 if 'fake' in img_name.lower() else 0)

    print(f"\nProcessing {len(image_paths)} images...")

    all_results = []

    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        print(
            f"\nProcessing image {idx + 1}/{len(image_paths)}: {os.path.basename(img_path)}")

        # Load and preprocess image
        img_pil = Image.open(img_path).convert('RGB')
        img_np = np.array(img_pil.resize((224, 224)))

        # RGB input
        rgb_input = rgb_transform(img_pil).unsqueeze(0).to(device)

        # Wavelet input
        img_for_wavelet = np.array(img_pil.resize((224, 224))) / 255.0
        wavelet_packets = compute_wavelet_packets(
            img_for_wavelet.transpose(2, 0, 1),
            wavelet='haar',
            level=args.wavelet_level
        )
        wavelet_input = wavelet_packets.unsqueeze(0).to(device)

        # Generate Grad-CAMs
        rgb_cam, rgb_pred = rgb_gradcam.generate_cam(rgb_input)
        wavelet_cam, wavelet_pred = wavelet_gradcam.generate_cam(wavelet_input)

        # Fusion model prediction
        fusion_pred = None
        if fusion_model is not None:
            with torch.no_grad():
                rgb_embed = rgb_embed_model(rgb_input)
                wavelet_embed = wavelet_embed_model(wavelet_input)
                fusion_pred = fusion_model(rgb_embed, wavelet_embed)

        # Create overlay
        rgb_overlay, _ = overlay_cam_on_image(img_np, rgb_cam)

        # For wavelet CAM, resize to match image
        wavelet_cam_resized = cv2.resize(wavelet_cam, (224, 224))
        wavelet_heatmap = cv2.applyColorMap(
            np.uint8(255 * wavelet_cam_resized), cv2.COLORMAP_JET)
        wavelet_heatmap = cv2.cvtColor(
            wavelet_heatmap, cv2.COLOR_BGR2RGB) / 255.0
        wavelet_overlay = 0.5 * wavelet_heatmap + 0.5 * (img_np / 255.0)
        wavelet_overlay = np.clip(wavelet_overlay, 0, 1)

        # Wavelet energy visualization
        wavelet_energy = visualize_wavelet_packets(
            wavelet_packets, args.wavelet_level)

        # Store results
        result = {
            'original': img_np,
            'rgb_cam': rgb_overlay,
            'wavelet_cam': wavelet_overlay,
            'wavelet_vis': wavelet_energy,
            'label': label,
            'rgb_pred': rgb_pred,
            'wavelet_pred': wavelet_pred,
            'fusion_pred': fusion_pred,
            'img_name': os.path.basename(img_path)
        }
        all_results.append(result)

        # Save individual detailed figure
        predictions = {
            'rgb_prob': torch.sigmoid(rgb_pred).item(),
            'wavelet_prob': torch.sigmoid(wavelet_pred).item(),
            'fusion_prob': torch.sigmoid(fusion_pred).item() if fusion_pred is not None else None,
            'wavelet_cam_raw': wavelet_cam_resized
        }

        individual_path = os.path.join(
            args.output_dir,
            f'gradcam_detail_{idx:02d}_{os.path.splitext(result["img_name"])[0]}.png'
        )
        create_detailed_cam_figure(
            img_np, rgb_cam, wavelet_overlay, wavelet_energy,
            predictions, label, individual_path
        )

    # Create comparison figure
    comparison_path = os.path.join(args.output_dir, 'gradcam_comparison.png')
    create_comparison_figure(all_results, comparison_path)

    # Cleanup
    rgb_gradcam.remove_hooks()
    wavelet_gradcam.remove_hooks()

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for result in all_results:
        rgb_prob = torch.sigmoid(result['rgb_pred']).item()
        wav_prob = torch.sigmoid(result['wavelet_pred']).item()
        fus_prob = torch.sigmoid(result['fusion_pred']).item(
        ) if result['fusion_pred'] is not None else None

        true_label = 'REAL' if result['label'] == 1 else 'FAKE'
        rgb_correct = (rgb_prob > 0.5) == result['label']
        wav_correct = (wav_prob > 0.5) == result['label']
        fus_correct = (
            fus_prob > 0.5) == result['label'] if fus_prob is not None else None

        print(f"\n{result['img_name']}:")
        print(f"  True: {true_label}")
        print(f"  RGB:     {rgb_prob:.3f} ({'✓' if rgb_correct else '✗'})")
        print(f"  Wavelet: {wav_prob:.3f} ({'✓' if wav_correct else '✗'})")
        if fus_prob is not None:
            print(f"  Fusion:  {fus_prob:.3f} ({'✓' if fus_correct else '✗'})")

    print(f"\n\nResults saved to: {args.output_dir}")


def main():
    opt = tSNE_Options().parse()
    process_images(opt)


if __name__ == '__main__':
    main()
