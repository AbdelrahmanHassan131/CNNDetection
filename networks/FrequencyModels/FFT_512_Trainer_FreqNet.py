import torch
import torch.nn as nn
from networks.FrequencyModels.FreqNet.FreqNet_DeepfakeDetection.networks.freqnet import FreqNet
from networks.base_model import BaseModel


class FFT_512_Trainer_WithFreqNet(BaseModel):
    def name(self):
        return 'FFT_512_Trainer_WithFreqNet'

    def __init__(self, opt):
        super(FFT_512_Trainer_WithFreqNet, self).__init__(opt)

        # === Device setup ===
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count()
        print(f"Using device: {self.device}, GPUs available: {self.gpu_count}")

        # === Backbone: FreqNet ===
        self.backbone = FreqNet(num_classes=4).to(self.device)
        if not self.isTrain or opt.continue_train:
            checkpoint_path = "networks/FrequencyModels/FreqNet/FreqNet_DeepfakeDetection/4-classes-freqnet-v2.pth"
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True)
            self.backbone.load_state_dict(checkpoint, strict=False)

        # === Determine feature dimension dynamically ===
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512).to(self.device)
            features = self.backbone.forward_features(dummy) if hasattr(
                self.backbone, 'forward_features') else self.backbone(dummy)
            feat_dim = features.view(1, -1).shape[1]

        # === Replace classifier head ===
        self.new_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)  # Binary classification
        ).to(self.device)

        # === Initialize new layers if starting fresh ===
        if self.isTrain and not opt.continue_train:
            nn.init.normal_(self.new_head[0].weight, 0.0, opt.init_gain)
            nn.init.normal_(self.new_head[2].weight, 0.0, opt.init_gain)

        # === Combine backbone and head ===
        self.model = nn.Sequential(
            self.backbone, nn.Flatten(), self.new_head).to(self.device)

        # === Multi-GPU support ===
        if self.gpu_count > 1:
            self.model = nn.DataParallel(self.model)
            print(f"DataParallel enabled on {self.gpu_count} GPUs")

        # === Loss and optimizer ===
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # ✅ Required for BaseModel save/load
        self.model_names = ['model']

    def set_input(self, input_data):
        """Supports tuple/list or dict input"""
        if isinstance(input_data, (list, tuple)):
            image, label = input_data
        else:
            image = input_data['image']
            label = input_data['label']

        self.input = image.to(self.device)
        self.label = label.float().to(self.device)

    def forward(self, x):
        """Forward pass through combined model"""
        return self.model(x)

    def optimize_parameters(self):
        logits = self.forward(self.input)
        loss = self.loss_fn(logits.squeeze(1), self.label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss
