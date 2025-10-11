import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from networks.resnet import resnet50
from networks.base_model import BaseModel


class FFT_512_Trainer(BaseModel):
    def name(self):
        return 'FFT_512_Trainer'

    def __init__(self, opt):
        super(FFT_512_Trainer, self).__init__(opt)

        # === MODEL SETUP ===
        pretrained_flag = self.isTrain and not opt.continue_train
        self.model = resnet50(pretrained=pretrained_flag)

        # Replace final FC with same head (2048 → 512 → 1)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Initialize weights (same as Wang2020)
        if self.isTrain and not opt.continue_train:
            torch.nn.init.normal_(
                self.model.fc[0].weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(
                self.model.fc[2].weight.data, 0.0, opt.init_gain)

        # === OPTIMIZER & LOSS ===
        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Load pretrained if needed
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(opt.gpu_ids[0])

        # === SAVE DIRECTORY FOR FFT IMAGES ===
        self.save_fft_dir = os.path.join(
            opt.checkpoints_dir, opt.name, "fft_debug_images")
        os.makedirs(self.save_fft_dir, exist_ok=True)

    # === HELPER FUNCTIONS ===
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    # === FFT PREPROCESSING ===
    def fft_preprocess(self, x):
        """
        Applies 2D FFT to each RGB channel separately,
        then takes the log magnitude spectrum.
        Also saves debug images.
        """
        # Safety check: label must be available
        if not hasattr(self, "label") or self.label is None:
            raise RuntimeError(
                "❌ Label not found for current batch. Stopping training!")

        # FFT
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)

        # Magnitude spectrum (avoid log(0))
        magnitude = torch.sqrt(fft_shift.real ** 2 +
                               fft_shift.imag ** 2 + 1e-8)
        log_mag = torch.log1p(magnitude)

        # Normalize per image to [0, 1]
        log_mag = log_mag - log_mag.amin(dim=(2, 3), keepdim=True)
        log_mag = log_mag / (log_mag.amax(dim=(2, 3), keepdim=True) + 1e-8)

        # Save FFT images
        self._save_fft_images(log_mag)

        return log_mag

    def _save_fft_images(self, log_mag):
        """
        Saves each FFT preprocessed image to a fixed directory
        depending on its label (0 or 1), keeping filename structure.
        """
        base_dir = r"G:\Master's Of Science Computer Engineering\thesis deepfake detection\fourth task\training images\FFT"
        batch_size = log_mag.size(0)

        for i in range(batch_size):
            try:
                label_value = int(self.label[i].item())
            except Exception as e:
                raise RuntimeError(
                    f"❌ Failed to load label for image {i}: {e}"
                )

            # choose save directory based on label
            if label_value == 0:
                save_dir = os.path.join(base_dir, "0")
            elif label_value == 1:
                save_dir = os.path.join(base_dir, "1")
            else:
                raise ValueError(
                    f"Unexpected label value {label_value} — expected 0 or 1"
                )

            # ensure directory exists
            os.makedirs(save_dir, exist_ok=True)

            # create filename (same as before)
            filename = f"fft_img_{self.total_steps if hasattr(self, 'total_steps') else 0}_batch_{i}_label_{label_value}.png"
            save_path = os.path.join(save_dir, filename)

            # save image
            vutils.save_image(log_mag[i], save_path)

    # === FORWARD / TRAINING ===
    def forward(self):
        fft_input = self.fft_preprocess(self.input)
        self.output = self.model(fft_input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
