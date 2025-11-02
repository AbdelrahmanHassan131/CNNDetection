import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from networks.base_model import BaseModel
import math


# -------------------------
# Torch-based Haar single-level DWT (GPU friendly)
# -------------------------
def haar_dwt_2d_tensor(x):
    """
    Input:
        x: (B,3,H,W) float tensor
    Output:
        y: (B,12,H, W) -> we compute 12 subbands per sample (4 per channel),
        but we produce them by doing conv stride=2 and then upsampling back to (H,W)
        so the final tensor is same spatial size as input (matching behavior in your code).
    Implementation details:
        - Use grouped conv with predefined 2x2 Haar filters for each channel.
        - conv with stride=2 downsamples to H/2,W/2 producing 12 channels (4 per input channel).
        - then upsample back to original H,W using bilinear interpolation.
    """
    assert x.ndim == 4 and x.size(1) == 3, "Expected (B,3,H,W)"
    device = x.device
    dtype = x.dtype

    # Haar 2x2 filters: LL, LH, HL, HH
    # Note: normalization factor 0.5 to preserve energy
    LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype, device=device)
    LH = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=dtype, device=device)
    HL = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=dtype, device=device)
    HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=dtype, device=device)

    # Build weight: out_channels = 12 (4 subbands * 3 channels), groups=3
    # Weight shape for conv2d with groups=3: (out_channels, in_channels_per_group, kH, kW)
    # in_channels_per_group = 1
    filters = torch.stack([LL, LH, HL, HH], dim=0)  # (4,2,2)
    # Repeat filters for each input channel
    filters = filters.unsqueeze(1)  # (4,1,2,2)
    filters = filters.repeat(3, 1, 1, 1)  # (12,1,2,2)

    # conv2d grouped: in_channels=3, out_channels=12, groups=3
    # Ensure dtype and device match
    weight = filters.to(dtype=dtype, device=device)

    # Perform grouped conv
    # x: (B,3,H,W) -> out: (B,12,H/2,W/2)
    out = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=3)

    # upsample back to original size (B,12,H,W)
    H, W = x.shape[2], x.shape[3]
    out_up = F.interpolate(out, size=(
        H, W), mode='bilinear', align_corners=False)

    # normalize per-sample to reduce scale differences
    # compute mean/std across channels & spatial dims per sample
    B = out_up.shape[0]
    out_flat = out_up.view(B, -1)
    mean = out_flat.mean(dim=1).view(B, 1, 1, 1)
    std = out_flat.std(dim=1).view(B, 1, 1, 1).clamp(min=1e-6)
    out_norm = (out_up - mean) / std

    return out_norm


# -------------------------
# Wavelet embedding CNN: 12 -> 3 (learnable)
# -------------------------
class WaveletEmbedCNN(nn.Module):
    def __init__(self, in_ch=12, out_ch=3):
        super(WaveletEmbedCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1, padding=0),
            nn.Tanh()  # keep outputs in reasonable range
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Full trainer (uses BaseModel)
# -------------------------
class Wavelet_ResNet_Trainer(BaseModel):
    def name(self):
        return "Wavelet_ResNet50_Trainer"

    def __init__(self, opt):
        super(Wavelet_ResNet_Trainer, self).__init__(opt)
        device = self.device
        print(f"\n🔧 Building Wavelet-ResNet50 Model on {device}...")

        # components (do not move to device yet)
        self.wavelet_embed = WaveletEmbedCNN(in_ch=12, out_ch=3)

        # load pretrained resnet50 backbone
        try:
            from torchvision.models import ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            self.backbone = resnet50(pretrained=True)
        # remove final fc
        self.backbone.fc = nn.Identity()

        # determine output dimension (on CPU dummy)
        self.wavelet_embed.eval()
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 12, 224, 224)
            feat = self.backbone(self.wavelet_embed(dummy))
            feat_dim = feat.shape[1]
            print(f"   🔍 Detected feature dimension: {feat_dim}")

        # classification head
        self.fc_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        # assemble model: wavelet_embed -> backbone -> flatten -> head
        # make a single nn.Module for wrapping
        class _Model(nn.Module):
            def __init__(self, embed, backbone, head):
                super().__init__()
                self.embed = embed
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                # x expected (B,3,H,W)
                # compute dwt features
                wav = haar_dwt_2d_tensor(x)   # (B,12,H,W) on same device
                mapped = self.embed(wav)      # (B,3,H,W)
                feats = self.backbone(mapped)  # (B, feat_dim)
                out = self.head(feats)
                return out

        self.model = _Model(self.wavelet_embed, self.backbone, self.fc_head)

        # move model to device before DataParallel
        print(f"   📍 Moving model to {device} ...")
        self.model.to(device)

        # DataParallel if requested
        if getattr(opt, 'gpu_ids', None) and len(opt.gpu_ids) > 1 and torch.cuda.device_count() > 1:
            print(f"   🎮 Enabling DataParallel on GPUs: {opt.gpu_ids}")
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
            print("   ✓ Model replicated across GPUs")
        else:
            print("   ℹ️ Single GPU / CPU mode")

        # Training components
        if self.isTrain:
            print("\n⚙️ Initializing training components...")
            # initialize head weights

            def _init_head(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.fc_head.apply(_init_head)

            # loss: BCEWithLogits (we will set pos_weight later if balancing)
            self.loss_fn = nn.BCEWithLogitsLoss()

            # optimizer: discriminative LR - smaller for backbone
            backbone_params = list(self.backbone.parameters())
            head_params = list(self.wavelet_embed.parameters()
                               ) + list(self.fc_head.parameters())
            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': opt.lr * 0.1},
                {'params': head_params, 'lr': opt.lr}
            ], betas=(opt.beta1, 0.999), weight_decay=1e-4)

            # scheduler (monitor AP/acc externally)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-7
            )

            # bookkeeping for class balancing
            self.use_weighted_loss = getattr(opt, 'class_bal', False)
            self.pos_weight_computed = False
            self._label_counter = {'0': 0, '1': 0}

        # load checkpoint if required
        if getattr(opt, 'continue_train', False):
            try:
                self.load_networks(opt.epoch)
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")

        print("✅ Model setup complete!\n")

    # set_input: accept torch tensor (B,C,H,W) or numpy/PIL per-sample
    def set_input(self, input_data):
        # flexible input formats
        if isinstance(input_data, (list, tuple)):
            image, label = input_data
        elif isinstance(input_data, dict):
            image = input_data.get('image')
            label = input_data.get('label')
        else:
            raise ValueError("Unsupported input data type")

        # if image is numpy array per-sample, convert; if torch tensor, keep
        # Expect image to be torch.Tensor shaped (B,3,H,W) or list of tensors
        if isinstance(image, torch.Tensor):
            imgs = image
        else:
            # image could be list/ndarray - convert to tensor
            imgs = torch.stack([torch.tensor(img).permute(2, 0, 1).float() if (isinstance(img, (list, tuple)) or (
                hasattr(img, 'ndim') and img.ndim == 3)) else torch.tensor(img).float() for img in image])

        # Move to device, ensure contiguous & normalized by user pipeline
        imgs = imgs.to(self.device, non_blocking=True).contiguous()
        self.input = imgs
        self.label = torch.tensor(
            label, dtype=torch.float32, device=self.device).contiguous()

        # update counters
        self._label_counter['0'] += int((self.label == 0).sum().item())
        self._label_counter['1'] += int((self.label == 1).sum().item())

    def forward(self, x):
        # ensure contiguous and on correct device
        x = x.to(self.device, non_blocking=True).contiguous()
        return self.model(x)

    def optimize_parameters(self):
        self.model.train()

        # compute pos_weight when we have enough samples
        if self.use_weighted_loss and (not self.pos_weight_computed):
            total = self._label_counter['0'] + self._label_counter['1']
            if total >= 1000 and self._label_counter['1'] > 0:
                pos_weight = torch.tensor(
                    [self._label_counter['0'] / float(max(1, self._label_counter['1']))], device=self.device)
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(
                    f"\n🎯 Class balancing enabled! Fake={self._label_counter['0']}, Real={self._label_counter['1']}, pos_weight={pos_weight.item():.4f}\n")
                self.pos_weight_computed = True

        logits = self.forward(self.input)  # (B,1)
        loss = self.loss_fn(logits.view(-1), self.label.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.loss = loss.detach().cpu().item()

        # compute batch acc
        with torch.no_grad():
            probs = torch.sigmoid(logits.view(-1))
            preds = (probs > 0.5).float()
            self.batch_acc = ((preds == self.label.view(-1)
                               ).float().mean()).cpu().item()

    def update_learning_rate(self, val_metric):
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.step(val_metric)

    def print_label_stats(self):
        total = self._label_counter['0'] + self._label_counter['1']
        if total > 0:
            print(
                f"📊 Label distribution so far: Real={self._label_counter['1']} ({100*self._label_counter['1']/total:.1f}%), Fake={self._label_counter['0']} ({100*self._label_counter['0']/total:.1f}%)")
