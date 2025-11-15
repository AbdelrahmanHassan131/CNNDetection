import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel, init_weights


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module to fuse embeddings from two different models.
    Allows one embedding to attend to the other.
    """

    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value):
        """
        Args:
            query: [B, embed_dim] - embeddings from one model
            key_value: [B, embed_dim] - embeddings from another model
        Returns:
            [B, embed_dim] - attended features
        """
        # Add sequence dimension for attention: [B, 1, embed_dim]
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)

        # Cross attention
        attn_output, _ = self.multihead_attn(query, key_value, key_value)

        # Remove sequence dimension: [B, embed_dim]
        attn_output = attn_output.squeeze(1)
        query = query.squeeze(1)

        # Residual connection and normalization
        x = self.norm1(query + attn_output)

        # Feed-forward network with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class MHAFusionClassifier(nn.Module):
    """
    Multi-Head Attention Fusion model for combining RGB and Wavelet models.
    """

    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1, fusion_type='cross_attention'):
        super(MHAFusionClassifier, self).__init__()

        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == 'cross_attention':
            # Cross-attention: RGB attends to Wavelet and vice versa
            self.rgb_to_wavelet_attn = CrossAttentionFusion(
                embed_dim, num_heads, dropout)
            self.wavelet_to_rgb_attn = CrossAttentionFusion(
                embed_dim, num_heads, dropout)

            # Combine attended features
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'self_attention':
            # Concatenate and apply self-attention
            self.input_projection = nn.Linear(embed_dim * 2, embed_dim)

            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            self.norm = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )

        elif fusion_type == 'concat':
            # Simple concatenation with MLP
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, rgb_embed, wavelet_embed):
        """
        Args:
            rgb_embed: [B, 128] - embeddings from RGB model
            wavelet_embed: [B, 128] - embeddings from Wavelet model
        Returns:
            [B, 1] - logits for binary classification
        """
        if self.fusion_type == 'cross_attention':
            # RGB attends to Wavelet
            rgb_attended = self.rgb_to_wavelet_attn(rgb_embed, wavelet_embed)

            # Wavelet attends to RGB
            wavelet_attended = self.wavelet_to_rgb_attn(
                wavelet_embed, rgb_embed)

            # Concatenate attended features
            fused = torch.cat([rgb_attended, wavelet_attended], dim=1)
            fused = self.fusion_layer(fused)

        elif self.fusion_type == 'self_attention':
            # Concatenate and project
            concat = torch.cat([rgb_embed, wavelet_embed], dim=1)
            x = self.input_projection(concat)

            # Add sequence dimension
            x = x.unsqueeze(1)  # [B, 1, embed_dim]

            # Self-attention
            attn_output, _ = self.self_attn(x, x, x)
            attn_output = attn_output.squeeze(1)

            # Residual and FFN
            x = x.squeeze(1)
            x = self.norm(x + attn_output)
            fused = x + self.ffn(x)

        else:  # concat
            concat = torch.cat([rgb_embed, wavelet_embed], dim=1)
            fused = self.fusion_layer(concat)

        # Final classification
        output = self.classifier(fused)

        return output


class MHAFusionTrainer(BaseModel):
    """
    Trainer for Multi-Head Attention Fusion model.
    Combines pre-trained RGB (Wang2020) and Wavelet (Wolter2022) models.
    """

    def name(self):
        return 'MHAFusion'

    def __init__(self, opt):
        super(MHAFusionTrainer, self).__init__(opt)

        # Fusion parameters
        self.embed_dim = getattr(opt, 'embed_dim', 128)
        self.num_heads = getattr(opt, 'num_heads', 4)
        self.dropout = getattr(opt, 'dropout', 0.1)
        self.fusion_type = getattr(opt, 'fusion_type', 'cross_attention')
        self.freeze_base_models = getattr(opt, 'freeze_base_models', True)

        # Paths to pre-trained models
        self.rgb_model_path = opt.rgb_model_path
        self.wavelet_model_path = opt.wavelet_model_path

        # Load pre-trained RGB model (Wang2020)
        print(f"Loading RGB model from: {self.rgb_model_path}")
        self.rgb_model = self._load_rgb_model(opt)

        # Load pre-trained Wavelet model (Wolter2022)
        print(f"Loading Wavelet model from: {self.wavelet_model_path}")
        self.wavelet_model = self._load_wavelet_model(opt)

        # Freeze base models if specified
        if self.freeze_base_models:
            print("Freezing base models (RGB and Wavelet)")
            self._freeze_model(self.rgb_model)
            self._freeze_model(self.wavelet_model)

        # Create fusion model
        if self.isTrain and not opt.continue_train:
            self.model = MHAFusionClassifier(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                fusion_type=self.fusion_type
            )
            init_weights(self.model, init_type='xavier', gain=opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = MHAFusionClassifier(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                fusion_type=self.fusion_type
            )

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Optimizer for fusion model only (base models frozen)
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=1e-4
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=opt.lr,
                    momentum=0.9,
                    weight_decay=1e-4
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Load checkpoint if continuing training or evaluating
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        # Move all models to device
        self.rgb_model.to(self.device)
        self.wavelet_model.to(self.device)
        self.model.to(self.device)

        print(f"MHA Fusion model created with {self.fusion_type} fusion")
        print(f"Base models frozen: {self.freeze_base_models}")

    def _load_rgb_model(self, opt):
        """Load pre-trained RGB model (Wang2020 ResNet50)"""
        from networks.resnet import resnet50

        model = resnet50(num_classes=1)
        state_dict = torch.load(self.rgb_model_path, map_location='cuda')

        # The saved model has fc as Sequential: [Linear(2048->128), ReLU, Dropout, Linear(128->1)]
        # We need to modify the architecture to match before loading
        # Check if fc is Sequential in the state_dict
        if 'fc.0.weight' in state_dict['model']:
            # The saved model has fc as Sequential
            # Temporarily replace fc with Sequential to load weights
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )

        model.load_state_dict(state_dict['model'])

        # Now remove the final classification layer to get 128-dim embeddings
        # Keep only: Linear(2048->128) -> ReLU
        in_features = model.fc[0].in_features if isinstance(
            model.fc, nn.Sequential) else model.fc.in_features
        model.fc = nn.Sequential(
            model.fc[0],  # Linear(2048 -> 128)
            model.fc[1],  # ReLU
            # Remove Dropout and final Linear layer
        )

        model.eval()
        return model

    def _load_wavelet_model(self, opt):
        """Load pre-trained Wavelet model (Wolter2022)"""
        from networks.FrequencyModels.WaveletsPacketsScratch_128.Trainer_WaveletsPacketsScratch_128 import (
            WaveletPacketCNN
        )

        # Calculate input channels for wavelet model
        wavelet_level = getattr(opt, 'wavelet_level', 3)
        num_packets_per_channel = 4 ** wavelet_level
        input_channels = 3 * num_packets_per_channel

        model = WaveletPacketCNN(input_channels=input_channels, num_classes=1)
        state_dict = torch.load(self.wavelet_model_path, map_location='cuda')
        model.load_state_dict(state_dict['model'])

        # Remove the final classification layer
        # WaveletPacketCNN structure: ... -> classifier = Sequential[
        #   Linear(512, 128), ReLU, Dropout, Linear(128, 1)
        # ]
        # We want to stop after Linear(512, 128) -> ReLU (128-dim embeddings)

        # Check the classifier structure
        if isinstance(model.classifier, nn.Sequential):
            # Keep only the first two layers: Linear(512->128) and ReLU
            model.classifier = nn.Sequential(
                model.classifier[0],  # Linear(512, 128)
                model.classifier[1],  # ReLU
                # Remove Dropout and final Linear layer
            )
        else:
            raise ValueError(
                "Unexpected classifier structure in WaveletPacketCNN")

        model.eval()
        return model

    def _freeze_model(self, model):
        """Freeze all parameters of a model"""
        for param in model.parameters():
            param.requires_grad = False

    def adjust_learning_rate(self, min_lr=1e-6):
        """Reduce learning rate by a factor of 10"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def __call__(self, rgb_input, wavelet_input):
        """
        Make trainer callable for validation.

        Args:
            rgb_input: [B, 3, H, W] - RGB images
            wavelet_input: [B, 192, H, W] - Wavelet packet coefficients
        Returns:
            output logits [B, 1]
        """
        rgb_input = rgb_input.to(self.device)
        wavelet_input = wavelet_input.to(self.device)

        with torch.no_grad():
            # Extract embeddings from base models
            rgb_embed = self.rgb_model(rgb_input)  # [B, 128]
            wavelet_embed = self.wavelet_model(wavelet_input)  # [B, 128]

        # Fusion model forward pass
        output = self.model(rgb_embed, wavelet_embed)

        return output

    def train(self):
        """Set model to training mode"""
        self.model.train()
        # Keep base models in eval mode if frozen
        if self.freeze_base_models:
            self.rgb_model.eval()
            self.wavelet_model.eval()

    def eval(self):
        """Set all models to evaluation mode"""
        self.model.eval()
        self.rgb_model.eval()
        self.wavelet_model.eval()

    def set_input(self, input):
        """
        Process input from dataloader.

        Args:
            input: tuple of (rgb_images, wavelet_packets, labels)
                rgb_images: [B, 3, H, W]
                wavelet_packets: [B, 192, H, W]
                labels: [B]
        """
        rgb_imgs, wavelet_imgs, labels = input[0], input[1], input[2]

        self.rgb_input = rgb_imgs.to(self.device)
        self.wavelet_input = wavelet_imgs.to(self.device)
        self.label = labels.to(self.device).float()

    def forward(self):
        """Forward pass through all models"""
        # Extract embeddings from base models (no gradient)
        with torch.no_grad():
            self.rgb_embed = self.rgb_model(self.rgb_input)
            self.wavelet_embed = self.wavelet_model(self.wavelet_input)

        # Fusion model forward (with gradient)
        self.output = self.model(self.rgb_embed, self.wavelet_embed)

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
