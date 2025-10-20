import torch
import torch.nn as nn
from networks.FrequencyModels.FreqNet.FreqNet_DeepfakeDetection.networks.freqnet import FreqNet
from networks.base_model import BaseModel


class FFT_512_Trainer_WithFreqNet(BaseModel):
    def name(self):
        return 'FFT_512_Trainer_WithFreqNet'

    def __init__(self, opt):
        super(FFT_512_Trainer_WithFreqNet, self).__init__(opt)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === Backbone: FreqNet ===
        self.backbone = FreqNet(num_classes=4)

        # === Load pretrained weights ===
        checkpoint_path = r"networks\FrequencyModels\FreqNet\FreqNet_DeepfakeDetection\4-classes-freqnet-v2.pth"
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True)
        self.backbone.load_state_dict(checkpoint, strict=False)

        self.backbone = self.backbone.to(device)

        # === Determine feature dimension dynamically ===
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512).to(device)
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(dummy)
            else:
                features = self.backbone(dummy)
            feat_dim = features.view(1, -1).shape[1]

        # === Replace classifier head ===
        self.new_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)  # Binary classification
        )

        # === Combine everything into one model for saving/loading ===
        self.model = nn.Sequential(self.backbone, nn.Flatten(), self.new_head)

        # === Initialize new layers ===
        if self.isTrain and not opt.continue_train:
            nn.init.normal_(self.new_head[0].weight, 0.0, opt.init_gain)
            nn.init.normal_(self.new_head[2].weight, 0.0, opt.init_gain)

        # === Loss & optimizer ===
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )

        self.model = self.model.to(device)
        self.device = device
        # ✅ Required for base_model saving/loading
        self.model_names = ['model']

    # ✅ Supports both dict or tuple input
    def set_input(self, input_data):
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
