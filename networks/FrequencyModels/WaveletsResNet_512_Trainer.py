import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from networks.base_model import BaseModel
import pywt
import cv2
import numpy as np

# =========================
# Wavelet preprocessing (FIXED)
# =========================


def rgb_to_wavelet12(img):
    """
    Convert RGB image (H,W,3) or (3,H,W) to 12-channel wavelet tensor (12,H,W)
    🆕 FIXED: Added proper normalization
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

    # 🆕 CRITICAL FIX: Normalize wavelet coefficients
    wavelet_mean = wavelet_resized.mean()
    wavelet_std = wavelet_resized.std() + 1e-8
    wavelet_resized = (wavelet_resized - wavelet_mean) / wavelet_std

    return torch.tensor(wavelet_resized, dtype=torch.float32)


# =========================
# Improved Wavelet Embedding CNN
# =========================

class WaveletEmbedCNN(nn.Module):
    def __init__(self, in_ch=12, out_ch=3):
        super(WaveletEmbedCNN, self).__init__()
        # 🆕 IMPROVED: Deeper network
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.embed(x)


# =========================
# Wavelet-ResNet50 Trainer (MULTI-GPU FIXED)
# =========================

class Wavelet_ResNet_Trainer(BaseModel):
    def name(self):
        return "Wavelet_ResNet50_Trainer"

    def __init__(self, opt):
        super(Wavelet_ResNet_Trainer, self).__init__(opt)

        # 🆕 CRITICAL FIX: Get device and ensure all components use it consistently
        device = self.device
        print(f"\n🔧 Building Wavelet-ResNet50 Model on {device}...")

        # === Build all components WITHOUT moving to device yet ===
        print("   📦 Creating wavelet embedding network...")
        self.wavelet_embed = WaveletEmbedCNN(in_ch=12, out_ch=3)

        # 🆕 CRITICAL FIX: Use weights parameter instead of pretrained
        print("   📦 Loading pretrained ResNet50 backbone...")
        try:
            from torchvision.models import ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except:
            # Fallback for older torchvision
            self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

        # === Determine feature dimension on CPU first ===
        print("   🔍 Detecting feature dimensions...")
        self.wavelet_embed.eval()
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 12, 224, 224)
            feat = self.backbone(self.wavelet_embed(dummy))
            feat_dim = feat.shape[1]
            print(f"   ✓ Feature dimension: {feat_dim}")

        # === Create classification head ===
        print("   📦 Creating classification head...")
        self.new_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        # 🆕 CRITICAL FIX: Assemble model THEN move everything to device at once
        print("   🔗 Assembling full model...")
        self.model = nn.Sequential(
            self.wavelet_embed,
            self.backbone,
            nn.Flatten(),
            self.new_head
        )

        # 🆕 CRITICAL FIX: Move entire model to device BEFORE DataParallel
        print(f"   📍 Moving model to {device}...")
        self.model = self.model.to(device)

        # 🆕 CRITICAL FIX: Now setup multi-GPU with properly placed model
        print("   🎮 Setting up GPU configuration...")
        if len(opt.gpu_ids) > 1 and torch.cuda.device_count() > 1:
            print(f"      🚀 Enabling DataParallel on GPUs: {opt.gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
            print(
                f"      ✓ Model will be replicated across {len(opt.gpu_ids)} GPUs")
        else:
            print(f"      ℹ️ Single GPU mode")

        # 🆕 BETTER INITIALIZATION
        if self.isTrain:
            print("\n⚙️ Initializing training components...")

            # Initialize new layers with Xavier
            if isinstance(self.model, nn.DataParallel):
                # Access the actual model inside DataParallel
                # The new_head is 4th component
                new_head = self.model.module[3]
            else:
                new_head = self.model[3]

            nn.init.xavier_normal_(new_head[1].weight)
            nn.init.zeros_(new_head[1].bias)
            nn.init.xavier_normal_(new_head[4].weight)
            nn.init.zeros_(new_head[4].bias)

            # Loss function with class balancing support
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.use_weighted_loss = opt.class_bal if hasattr(
                opt, 'class_bal') else False
            self.pos_weight_computed = False

            # 🆕 IMPROVED: Optimizer with weight decay
            print(f"   📊 Learning rate: {opt.lr:.2e}")
            print(f"   📊 Weight decay: 1e-4")
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=1e-4
            )

            # Learning rate scheduler
            print(f"   📊 Scheduler: ReduceLROnPlateau (patience=2)")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=2,
                verbose=True,
                min_lr=1e-7
            )

        # Load checkpoint if continuing training
        if opt.continue_train:
            try:
                print(
                    f"\n🔄 Continuing training — loading checkpoint from epoch {opt.epoch}...")
                self.load_networks(opt.epoch)
                print(
                    f"✅ Loaded checkpoint successfully (steps: {self.total_steps})")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")

        self.device = device
        self.model_names = ['model']
        print("✅ Model setup complete!\n")

    # =========================
    # Input preparation (IMPROVED)
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

        # Track label distribution for debugging
        if not hasattr(self, '_label_counter'):
            self._label_counter = {'0': 0, '1': 0}
        self._label_counter['0'] += (self.label == 0).sum().item()
        self._label_counter['1'] += (self.label == 1).sum().item()

    # =========================
    # Forward pass
    # =========================
    def forward(self, x):
        return self.model(x)

    # =========================
    # Training step (IMPROVED + WEIGHTED LOSS)
    # =========================
    def optimize_parameters(self):
        self.model.train()

        # Compute pos_weight after first few batches for class balancing
        if self.use_weighted_loss and not self.pos_weight_computed and hasattr(self, '_label_counter'):
            total = self._label_counter['0'] + self._label_counter['1']
            if total > 1000:  # Wait for 1000 samples
                n_neg = self._label_counter['0']
                n_pos = self._label_counter['1']
                if n_pos > 0 and n_neg > 0:
                    pos_weight = torch.tensor([n_neg / n_pos]).to(self.device)
                    self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    print(f"\n🎯 Class balancing enabled!")
                    print(f"   Fake (0): {n_neg} samples")
                    print(f"   Real (1): {n_pos} samples")
                    print(f"   Pos weight: {pos_weight.item():.4f}\n")
                    self.pos_weight_computed = True

        logits = self.forward(self.input)
        loss = self.loss_fn(logits.squeeze(1), self.label)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.loss = loss

        # Track predictions for debugging
        with torch.no_grad():
            preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).float()
            self.batch_acc = (preds == self.label).float().mean()

    # Method to update scheduler
    def update_learning_rate(self, val_acc):
        """Call this after validation to adjust learning rate"""
        if hasattr(self, 'scheduler'):
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            new_lr = self.optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"📉 Learning rate updated: {old_lr:.2e} → {new_lr:.2e}")

    # Method to print label distribution
    def print_label_stats(self):
        if hasattr(self, '_label_counter'):
            total = self._label_counter['0'] + self._label_counter['1']
            if total > 0:
                print(f"📊 Label distribution so far: "
                      f"Real={self._label_counter['1']} ({100*self._label_counter['1']/total:.1f}%), "
                      f"Fake={self._label_counter['0']} ({100*self._label_counter['0']/total:.1f}%)")
