import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from networks.base_model import BaseModel


# -------------------------
# GPU-friendly single-level Haar DWT (torch only)
# -------------------------
def haar_dwt_2d_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Compute a single-level Haar DWT in pure torch.
    Input:
        x: (B, 3, H, W), float32 on the correct device
    Output:
        out_norm: (B, 12, H, W) normalized per-sample
    Implementation:
        - grouped conv with 2x2 Haar filters to get 4 subbands per channel (B,12,H/2,W/2)
        - upsample to (H,W) to match ResNet expected spatial dims
        - per-sample normalization (mean/std)
    """
    assert x.dim() == 4 and x.size(1) == 3, "Input must be (B,3,H,W)"

    device = x.device
    dtype = x.dtype

    # Haar filters (2x2) normalized by 0.5 to preserve energy
    LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=dtype, device=device)
    LH = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=dtype, device=device)
    HL = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=dtype, device=device)
    HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=dtype, device=device)

    # Stack filters: (4, 1, 2, 2)
    filters = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)  # (4,1,2,2)
    # Repeat for each input channel -> (12,1,2,2)
    filters = filters.repeat(3, 1, 1, 1)

    # conv weight shape: (out_channels, in_channels/groups, kH, kW)
    weight = filters.to(dtype=dtype, device=device)

    # grouped conv: in_channels=3, out_channels=12, groups=3
    out = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=3)

    # upsample to original spatial resolution
    H, W = x.shape[2], x.shape[3]
    out_up = F.interpolate(out, size=(
        H, W), mode='bilinear', align_corners=False)

    # per-sample normalization (avoid division by zero)
    B = out_up.shape[0]
    flat = out_up.view(B, -1)
    mean = flat.mean(dim=1).view(B, 1, 1, 1)
    std = flat.std(dim=1).view(B, 1, 1, 1).clamp(min=1e-6)
    out_norm = (out_up - mean) / std

    return out_norm


# -------------------------
# Small learnable mapper: 12 -> 3
# -------------------------
class WaveletEmbedCNN(nn.Module):
    def __init__(self, in_ch: int = 12, out_ch: int = 3):
        super(WaveletEmbedCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1, padding=0, bias=True),
            nn.Tanh(),  # keeps values in a bounded range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Trainer
# -------------------------
class Wavelet_ResNet_Trainer(BaseModel):
    def name(self):
        return "Wavelet_ResNet50_Trainer"

    def __init__(self, opt):
        super(Wavelet_ResNet_Trainer, self).__init__(opt)
        self.opt = opt
        device = self.device
        print(f"\n🔧 Building Wavelet-ResNet50 Model on {device}...")

        # Components (instantiate on CPU first)
        self.wavelet_embed = WaveletEmbedCNN(in_ch=12, out_ch=3)

        # Load ResNet50 backbone (use weights API if available)
        try:
            from torchvision.models import ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            self.backbone = resnet50(pretrained=True)

        # Remove final fc; backbone outputs (B, 2048)
        self.backbone.fc = nn.Identity()

        # Probe feature dimension with CPU dummy (safe)
        self.wavelet_embed.eval()
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 12, 224, 224)
            feat = self.backbone(self.wavelet_embed(dummy))
            feat_dim = feat.shape[1]
            print(f"   🔍 Detected feature dimension: {feat_dim}")

        # Classification head
        self.fc_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

        # Build a small wrapper module to keep everything tidy
        class _Model(nn.Module):
            def __init__(self, embed, backbone, head):
                super(_Model, self).__init__()
                self.embed = embed
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                # x: (B,3,H,W) on same device as model
                # 1) Haar DWT (torch-only, GPU-friendly)
                wav = haar_dwt_2d_tensor(x)  # (B,12,H,W)
                # 2) learnable mapper 12->3
                mapped = self.embed(wav)     # (B,3,H,W)
                # 3) backbone and head
                feats = self.backbone(mapped)  # (B, feat_dim)
                out = self.head(feats)         # (B,1)
                return out

        # Instantiate wrapper (still on CPU)
        self.model = _Model(self.wavelet_embed, self.backbone, self.fc_head)

        # Move model to device BEFORE converting SyncBN / DataParallel
        self.model.to(device)

        # If multiple GPUs, convert to SyncBatchNorm then DataParallel (keeps BN safe)
        if getattr(opt, 'gpu_ids', None) and len(opt.gpu_ids) > 1 and torch.cuda.device_count() > 1:
            try:
                # Convert BatchNorm layers to SyncBatchNorm for correct multi-GPU BN behavior
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.model)
                print("   🔁 Converted BatchNorm -> SyncBatchNorm for multi-GPU")
            except Exception as e:
                print(f"   ⚠️ SyncBatchNorm conversion failed: {e}")

            # Wrap in DataParallel (we keep DataParallel for simplicity on Kaggle)
            self.model = nn.DataParallel(self.model, device_ids=opt.gpu_ids)
            print(f"   🎮 Enabled DataParallel on GPUs: {opt.gpu_ids}")

        else:
            print("   ℹ️ Single GPU / CPU mode")

        # Training utilities
        if self.isTrain:
            print("\n⚙️ Initializing training components...")

            # Initialize head weights
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.fc_head.apply(_init_weights)

            # Loss function (we'll set pos_weight dynamically later)
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Optimizer: discriminative LR (backbone smaller lr)
            backbone_params = list(self.backbone.parameters())
            head_params = list(self.wavelet_embed.parameters()
                               ) + list(self.fc_head.parameters())

            # Use AdamW for better generalization
            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': opt.lr * 0.1},
                {'params': head_params, 'lr': opt.lr}
            ], betas=(opt.beta1, 0.999), weight_decay=1e-4)

            # Scheduler: ReduceLROnPlateau monitoring validation metric (call scheduler.step(metric))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-7
            )

            # bookkeeping for weighted loss
            self.use_weighted_loss = getattr(opt, 'class_bal', False)
            self.pos_weight_computed = False
            self._label_counter = {'0': 0, '1': 0}

        # Continue training (optional)
        if getattr(opt, 'continue_train', False):
            try:
                self.load_networks(opt.epoch)
            except Exception as e:
                print(f"   ⚠️ Failed to load checkpoint: {e}")

        print("✅ Model setup complete!\n")

    # -------------------------
    # Input setting (no torch.tensor on CPU)
    # -------------------------
    def set_input(self, input_data):
        """
        Accepts:
            - input_data as tuple/list: (images, labels)
            - input_data as dict with keys 'image' and 'label'
        Expects images to be torch.Tensor (B,3,H,W) or a list/ndarray convertible to tensor.
        Labels may be torch Tensor or numpy array/list/scalar; we convert using torch.as_tensor
        and move to device — WITHOUT creating a CPU tensor via torch.tensor(...)
        """
        if isinstance(input_data, (list, tuple)):
            images, labels = input_data
        elif isinstance(input_data, dict):
            images = input_data.get('image')
            labels = input_data.get('label')
        else:
            raise ValueError("Unsupported input_data type in set_input")

        # If images is a list/ndarray, convert to tensor properly
        if not isinstance(images, torch.Tensor):
            # images likely a list of HWC numpy arrays or PIL; convert safely
            imgs = []
            for im in images:
                if isinstance(im, torch.Tensor):
                    imgs.append(im)
                else:
                    # im is numpy array (H,W,C) or PIL -> convert
                    arr = torch.as_tensor(im, dtype=torch.float32)
                    # if HWC -> CHW
                    if arr.ndim == 3 and arr.shape[2] in (1, 3):
                        arr = arr.permute(2, 0, 1)
                    imgs.append(arr)
            imgs = torch.stack(imgs, dim=0)
        else:
            imgs = images

        # Move images to device (non_blocking=True if pinned memory used)
        imgs = imgs.to(self.device, non_blocking=True).contiguous()
        self.input = imgs

        # Convert labels to tensor via as_tensor and move to device
        lbl = torch.as_tensor(labels, dtype=torch.float32)
        lbl = lbl.to(self.device, non_blocking=True).contiguous()
        # ensure shape (B,) for BCE loss
        if lbl.dim() == 2 and lbl.size(1) == 1:
            lbl = lbl.view(-1)
        self.label = lbl

        # update counters (on CPU small ints is fine)
        with torch.no_grad():
            self._label_counter['0'] += int((self.label == 0).sum().item())
            self._label_counter['1'] += int((self.label == 1).sum().item())

    # -------------------------
    # Forward wrapper
    # -------------------------
    def forward(self, x: torch.Tensor):
        # ensure contiguous on correct device
        x = x.to(self.device, non_blocking=True).contiguous()
        # call the model (DataParallel or plain)
        return self.model(x)

    # -------------------------
    # Optimization step
    # -------------------------
    def optimize_parameters(self):
        self.model.train()

        # compute pos_weight once we have enough samples
        if self.use_weighted_loss and (not self.pos_weight_computed):
            total = self._label_counter['0'] + self._label_counter['1']
            if total >= 1000 and self._label_counter['1'] > 0:
                pos_weight = torch.tensor([self._label_counter['0'] / float(
                    max(1, self._label_counter['1']))], dtype=torch.float32, device=self.device)
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                print(
                    f"\n🎯 Class balancing enabled! Fake={self._label_counter['0']} Real={self._label_counter['1']} pos_weight={pos_weight.item():.4f}\n")
                self.pos_weight_computed = True

        # forward
        logits = self.forward(self.input)   # (B,1)
        # ensure logits is tensor on device
        if isinstance(logits, tuple) or isinstance(logits, list):
            logits = logits[0]

        logits = logits.view(-1)
        labels = self.label.view(-1)

        loss = self.loss_fn(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # stash metrics
        self.loss = float(loss.detach().cpu().item())
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            self.batch_acc = float(
                (preds == labels).float().mean().cpu().item())

    # -------------------------
    # Scheduler helper
    # -------------------------
    def update_learning_rate(self, val_metric):
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            # Expectation: caller passes validation accuracy or AP
            try:
                self.scheduler.step(val_metric)
            except Exception as e:
                print(f"   ⚠️ Scheduler step failed: {e}")

    # -------------------------
    # Printing helper
    # -------------------------
    def print_label_stats(self):
        total = self._label_counter['0'] + self._label_counter['1']
        if total > 0:
            print(
                f"📊 Label distribution so far: Real={self._label_counter['1']} ({100*self._label_counter['1']/total:.1f}%), Fake={self._label_counter['0']} ({100*self._label_counter['0']/total:.1f}%)")
