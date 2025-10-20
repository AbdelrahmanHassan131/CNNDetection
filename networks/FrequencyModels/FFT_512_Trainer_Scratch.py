import os
import torch
import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
from .FFTNET_Scratch import FFTNet        # the new model above
import torchvision.utils as vutils


class FFT_512_Trainer(BaseModel):
    def name(self):
        return 'FFT_512_Trainer'

    def __init__(self, opt):
        super(FFT_512_Trainer, self).__init__(opt)

        # === MODEL SETUP ===
        self.model = FFTNet()

        # Weight initialization
        if self.isTrain and not opt.continue_train:
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, opt.init_gain)

        # === OPTIMIZER & LOSS ===
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Load pretrained weights (if continuing training)
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(opt.gpu_ids[0])

        # === SAVE DIRECTORY FOR FFT IMAGES ===
        self.save_fft_dir = os.path.join(
            opt.checkpoints_dir, opt.name, "fft_debug_images")
        os.makedirs(self.save_fft_dir, exist_ok=True)

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def fft_preprocess(self, x):
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.sqrt(fft_shift.real ** 2 +
                               fft_shift.imag ** 2 + 1e-8)
        log_mag = torch.log1p(magnitude)
        log_mag = log_mag - log_mag.amin(dim=(2, 3), keepdim=True)
        log_mag = log_mag / (log_mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
        # self._save_fft_images(log_mag)
        return log_mag

    def _save_fft_images(self, log_mag):
        base_dir = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\fourth task\training images\FFT"
        batch_size = log_mag.size(0)
        for i in range(batch_size):
            label_value = int(self.label[i].item())
            save_dir = os.path.join(base_dir, str(label_value))
            os.makedirs(save_dir, exist_ok=True)
            filename = f"fft_img_{self.total_steps if hasattr(self, 'total_steps') else 0}_batch_{i}_label_{label_value}.png"
            save_path = os.path.join(save_dir, filename)
            vutils.save_image(log_mag[i], save_path)

    def forward(self):
        fft_input = self.fft_preprocess(self.input)
        self.output = self.model(fft_input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
