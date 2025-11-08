import torch
import numpy as np
import pywt
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from io import BytesIO
from random import random, choice
import cv2
from scipy.ndimage import gaussian_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===========================
# Dataset Creation Functions
# ===========================

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    """
    Create dataset with wavelet preprocessing.
    """
    return WaveletBinaryDataset(opt, root)


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


# ===========================
# Data Augmentation Functions
# ===========================

def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS,
    'nearest': Image.NEAREST
}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


# ===========================
# Wavelet Transform Functions
# ===========================

def compute_wavelet_packet_coeffs(img, wavelet='haar', level=3, mode='reflect'):
    """
    Compute wavelet packet coefficients for an RGB image.

    Args:
        img: numpy array (H, W, 3) or torch tensor (3, H, W)
        wavelet: wavelet type (default: 'haar')
        level: decomposition level (default: 3)
        mode: signal extension mode (default: 'reflect')

    Returns:
        numpy array of shape (C, H', W') where C = 3 * 4^level
    """
    if torch.is_tensor(img):
        img = img.cpu().numpy()

    # Ensure image is (H, W, 3)
    if img.ndim == 3 and img.shape[0] == 3:
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
    # Shape: (3*4^level, H', W')
    all_packets = np.array(all_packets, dtype=np.float32)

    return all_packets


def log_scale_packets(packets, epsilon=1e-10):
    """
    Apply log-scaling to packet coefficients as in the paper.

    Args:
        packets: numpy array of packet coefficients
        epsilon: small value to avoid log(0)

    Returns:
        log-scaled packet coefficients
    """
    return np.sign(packets) * np.log(np.abs(packets) + epsilon)


# ===========================
# Wavelet Dataset
# ===========================

class WaveletBinaryDataset(datasets.ImageFolder):
    """
    Extended ImageFolder that computes wavelet packets on-the-fly.
    This happens in DataLoader workers (CPU, parallelized).
    """

    def __init__(self, opt, root):
        self.opt = opt

        # Wavelet parameters
        self.compute_wavelets = getattr(opt, 'compute_wavelets', True)
        self.wavelet_type = getattr(opt, 'wavelet_type', 'haar')
        self.wavelet_level = getattr(opt, 'wavelet_level', 3)
        self.wavelet_mode = getattr(opt, 'wavelet_mode', 'reflect')
        self.use_log_packets = getattr(opt, 'use_log_packets', True)

        # Build transforms
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)

        if not opt.isTrain and opt.no_resize:
            rz_func = transforms.Lambda(lambda img: img)
        else:
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

        # Standard image transforms (don't include ToTensor yet)
        self.image_transform = transforms.Compose([
            rz_func,
            transforms.Lambda(lambda img: data_augment(img, opt)),
            crop_func,
            flip_func,
        ])

        # Initialize ImageFolder with no transform (we'll apply it in __getitem__)
        super().__init__(root, transform=None)

    def __getitem__(self, index):
        """
        Load and process a single sample.

        Returns:
            If compute_wavelets=True: (wavelet_tensor, label)
            If compute_wavelets=False: (image_tensor, label)
        """
        path, target = self.samples[index]

        # Load image
        img = self.loader(path)

        # Apply augmentations (resize, crop, flip, blur, jpeg)
        img = self.transform(img)

        if self.compute_wavelets:
            # ✅ Compute wavelet packets (runs in DataLoader worker - CPU, parallelized!)
            img_array = np.array(img)

            wavelet_coeffs = compute_wavelet_packet_coeffs(
                img_array,
                wavelet=self.wavelet_type,
                level=self.wavelet_level,
                mode=self.wavelet_mode
            )

            # Apply log-scaling if specified
            if self.use_log_packets:
                wavelet_coeffs = log_scale_packets(wavelet_coeffs)

            # Convert to tensor
            sample = torch.from_numpy(wavelet_coeffs).float()
        else:
            # Standard RGB image processing
            sample = transforms.ToTensor()(img)
            sample = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(sample)

        return sample, target
