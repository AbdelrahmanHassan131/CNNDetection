
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from networks.base_model import BaseModel
import pywt
import cv2
import numpy as np


def rgb_to_wavelet12(img):
    """
    Convert RGB image (H,W,3) or (3,H,W) to 12-channel wavelet tensor (12,H,W)
    """
    if torch.is_tensor(img):
        img = img.cpu().numpy()

    if img.shape[0] == 3:  # (3,H,W)
        img = img.transpose(1, 2, 0)  # (H,W,3)

    wavelet_channels = []
    for c in range(3):
        coeffs2 = pywt.dwt2(img[:, :, c], 'haar')
        LL, (LH, HL, HH) = coeffs2
        wavelet_channels.extend([LL, LH, HL, HH])

    wavelet_channels = np.stack(wavelet_channels, axis=0)  # (12,H/2,W/2)

    # Upsample to original H,W
    H, W = img.shape[:2]
    wavelet_resized = []
    for i in range(wavelet_channels.shape[0]):
        ch = cv2.resize(wavelet_channels[i],
                        (W, H), interpolation=cv2.INTER_LINEAR)
        wavelet_resized.append(ch)
    wavelet_resized = np.stack(wavelet_resized, axis=0)

    return torch.tensor(wavelet_resized, dtype=torch.float32)

# =========================
# Small CNN to reduce 12->3 channels
# =========================


class WaveletEmbedCNN(nn.Module):
    def __init__(self, in_ch=12, out_ch=3):
        super(WaveletEmbedCNN, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),         # Added spatial dropout!
            nn.Conv2d(32, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.embed(x)

# =========================
# Wavelet-ResNet50 Trainer
# =========================


class Wavelet_ResNet_Trainer(BaseModel):
    def name(self):
        return "Wavelet_ResNet50_Trainer"

    def __init__(self, opt):
        super(Wavelet_ResNet_Trainer, self).__init__(opt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine if we should load pretrained weights
        pretrained_flag = self.isTrain and not opt.continue_train

        # === Wavelet embedder 12->3 channels ===
        self.wavelet_embed = WaveletEmbedCNN(in_ch=12, out_ch=3).to(device)

        # === ResNet50 backbone (USE PRETRAINED!) ===
        self.backbone = resnet50(pretrained=pretrained_flag)
        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()
        self.backbone = self.backbone.to(device)

        # === Determine feature dimension dynamically ===
        with torch.no_grad():
            dummy = torch.randn(1, 12, 224, 224).to(device)
            feat = self.backbone(self.wavelet_embed(dummy))
            feat_dim = feat.shape[1]

        # === New classifier head with DROPOUT and SMALLER size ===
        self.new_head = nn.Sequential(
            nn.Linear(feat_dim, 128),  # Reduced from 512 to 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),           # Added dropout for regularization
            nn.Linear(128, 1)          # Binary classification
        ).to(device)

        # === Combine model for saving/loading ===
        self.model = nn.Sequential(
            self.wavelet_embed,
            self.backbone,
            nn.Flatten(),
            self.new_head
        )

        # Initialize new layers only for brand new training
        if self.isTrain and not opt.continue_train:
            nn.init.normal_(self.new_head[0].weight, 0.0, opt.init_gain)
            nn.init.normal_(self.new_head[3].weight, 0.0, opt.init_gain)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # Add WEIGHT DECAY for regularization
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=1e-4  # L2 regularization
            )

        if opt.continue_train:
            try:
                print(
                    "🔄 Continuing training — loading latest checkpoint... opt epoch is ", opt.epoch)
                self.load_networks(opt.epoch)
                print(
                    f"✅ Loaded checkpoint successfully (steps: {self.total_steps})")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")

        self.device = device
        self.model_names = ['model']

    # =========================
    # Input preparation
    # =========================
    def set_input(self, input_data):
        if isinstance(input_data, (list, tuple)):
            image, label = input_data
        else:
            image = input_data['image']
            label = input_data['label']

        # Convert to 12-channel wavelets
        wavelet_imgs = []
        for img in image:
            wavelet_imgs.append(rgb_to_wavelet12(img))
        self.input = torch.stack(wavelet_imgs).to(self.device)
        self.label = label.float().to(self.device)

    # =========================
    # Forward pass
    # =========================
    def forward(self, x):
        return self.model(x)

    # =========================
    # Training step
    # =========================
    def optimize_parameters(self):
        logits = self.forward(self.input)
        loss = self.loss_fn(logits.squeeze(1), self.label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss
