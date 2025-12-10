import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from networks.base_model import BaseModel, init_weights


def compute_wavelet_packet_coeffs(img, wavelet='haar', level=3, mode='reflect'):
    """
    Compute wavelet packet coefficients for an RGB image.
    NOTE: This function is now primarily used in validation/inference.
    During training, wavelets are computed in the DataLoader.

    Args:
        img: torch tensor of shape (3, H, W) or numpy array (H, W, 3)
        wavelet: wavelet type (default: 'haar')
        level: decomposition level (default: 3)
        mode: signal extension mode (default: 'reflect')

    Returns:
        torch tensor of shape (C, H', W') where C = 3 * 4^level
    """
    if torch.is_tensor(img):
        img = img.cpu().numpy()

    # Ensure image is (H, W, 3)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    H, W, _ = img.shape
    all_packets = []

    # Generate all wavelet packet paths at the given level
    def get_paths(level):
        """Generate all wavelet packet paths at a given level"""
        if level == 0:
            return ['']
        paths = []
        prev_paths = get_paths(level - 1)
        for path in prev_paths:
            for letter in ['a', 'h', 'v', 'd']:
                paths.append(path + letter)
        return paths

    packet_paths = get_paths(level)

    # Process each color channel
    for c in range(3):
        channel = img[:, :, c]

        # Create wavelet packet decomposition
        wp = pywt.WaveletPacket2D(
            data=channel, wavelet=wavelet, mode=mode, maxlevel=level)

        for path in packet_paths:
            coeff = wp[path].data
            all_packets.append(coeff)

    # Stack all packets: 3 channels * 4^level packets per channel
    all_packets = np.array(all_packets)  # Shape: (3*4^level, H', W')

    # Convert to tensor
    packets_tensor = torch.tensor(all_packets, dtype=torch.float32)

    return packets_tensor


def log_scale_packets(packets, epsilon=1e-10):
    """
    Apply log-scaling to packet coefficients as in the paper.

    Args:
        packets: torch tensor of packet coefficients
        epsilon: small value to avoid log(0)

    Returns:
        log-scaled packet coefficients
    """
    return torch.sign(packets) * torch.log(torch.abs(packets) + epsilon)


class WaveletPacketCNN(nn.Module):
    """
    CNN architecture for wavelet packet-based deepfake detection.
    Based on the architecture from Wolter et al. 2022.

    The paper uses a relatively simple CNN without ImageNet pretraining
    to fairly evaluate the wavelet packet features.
    """

    def __init__(self, input_channels, num_classes=1):
        super(WaveletPacketCNN, self).__init__()

        # Convolutional layers
        # The paper uses a progressive architecture
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head with dropout for regularization
        # Last two layers: 512 -> 128 (embeddings) -> 1 (binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


class WolterWaveletPacketTrainer(BaseModel):
    """
    Wavelet-Packet DeepFake Detection Trainer
    Based on: Wolter et al. "Wavelet-Packets for Deepfake Image Analysis and Detection"
    Machine Learning, ECML PKDD 2022 Journal Track

    OPTIMIZED VERSION: Wavelets are computed in DataLoader workers (parallel CPU processing)
    """

    def name(self):
        return 'WolterWaveletPacket2022'

    def __init__(self, opt):
        super(WolterWaveletPacketTrainer, self).__init__(opt)

        # Wavelet packet parameters (for validation/inference only)
        self.wavelet_type = getattr(opt, 'wavelet_type', 'haar')
        self.wavelet_level = getattr(opt, 'wavelet_level', 3)
        self.wavelet_mode = getattr(opt, 'wavelet_mode', 'reflect')
        self.use_log_packets = getattr(opt, 'use_log_packets', True)

        # Calculate number of input channels
        # At level 3: 4^3 = 64 packets per channel
        # For RGB: 3 * 64 = 192 channels
        num_packets_per_channel = 4 ** self.wavelet_level
        input_channels = 3 * num_packets_per_channel

        # Determine if we should initialize from scratch or load
        if self.isTrain and not opt.continue_train:
            # Create model from scratch
            self.model = WaveletPacketCNN(
                input_channels=input_channels,
                num_classes=1
            )
            # Initialize weights using specified initialization
            init_weights(self.model, init_type='normal', gain=opt.init_gain)

        if not self.isTrain or opt.continue_train:
            # Create model architecture (will load weights later)
            self.model = WaveletPacketCNN(
                input_channels=input_channels,
                num_classes=1
            )

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Initialize optimizer with weight decay as in the paper
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=1e-4  # Regularization
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=opt.lr,
                    momentum=0.9,
                    weight_decay=1e-4  # Regularization
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Load checkpoint if continuing training or evaluating
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        """Reduce learning rate by a factor of 10"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def __call__(self, input_tensor):
        """
        Make trainer callable for validation.

        Args:
            input_tensor: 
                - If from dataloader with compute_wavelets=True: wavelet packets (B, 192, H, W)
                - If RGB images: (B, 3, H, W) - will compute wavelets
        Returns:
            output logits (B, 1)
        """
        # Check if input is already wavelet packets (192 channels) or RGB (3 channels)
        if input_tensor.shape[1] == 192:  # Already wavelet packets from dataloader
            # ✅ OPTIMIZED PATH: Direct forward pass, no preprocessing needed!
            output = self.model(input_tensor.to(self.device))
            return output

        # Fallback: Input is RGB, compute wavelets (for backward compatibility)
        batch_packets = []
        for img in input_tensor:
            # Compute wavelet packet coefficients
            packets = compute_wavelet_packet_coeffs(
                img,
                wavelet=self.wavelet_type,
                level=self.wavelet_level,
                mode=self.wavelet_mode
            )

            # Apply log-scaling if specified
            if self.use_log_packets:
                packets = log_scale_packets(packets)

            batch_packets.append(packets)

        # Stack into batch and run through model
        wavelet_input = torch.stack(batch_packets).to(self.device)
        output = self.model(wavelet_input)

        return output

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def set_input(self, input):
        """
        Process input from dataloader.

        OPTIMIZED: Input is already wavelet packets from dataloader!
        No wavelet computation needed here - just transfer to GPU.

        Args:
            input: tuple of (wavelet_packets, labels)
                wavelet_packets: torch tensor (B, 192, H, W) - already computed in dataloader
                labels: torch tensor (B,)
        """
        wavelet_packets, labels = input[0], input[1]

        # ✅ OPTIMIZED: Just transfer to GPU - wavelets already computed!
        self.input = wavelet_packets.to(self.device)
        self.label = labels.to(self.device).float()

    def forward(self):
        """Forward pass through the model"""
        self.output = self.model(self.input)

    def get_loss(self):
        """Calculate and return the loss"""
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        """Perform one optimization step"""
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
