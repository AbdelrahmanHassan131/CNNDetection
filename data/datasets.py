import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice, uniform
from io import BytesIO
from PIL import Image, ImageFile, ImageEnhance
from scipy.ndimage.filters import gaussian_filter

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------
# Helper functions (replacing lambdas)
# ----------------------------

def identity_transform(img):
    """Return the image unchanged."""
    return img


def apply_data_augment(img, opt):
    """Wrapper for data_augment to make it pickle-safe."""
    return data_augment(img, opt)


# ----------------------------
# Dataset creation
# ----------------------------

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def resize_with_opt(img, opt):
    return custom_resize(img, opt)


def augment_with_opt(img, opt):
    return data_augment(img, opt)


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(identity_transform)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(identity_transform)

    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(identity_transform)
    else:
        from functools import partial
        rz_func = transforms.Lambda(partial(resize_with_opt, opt=opt))

    dset = datasets.ImageFolder(
        root,
        transforms.Compose([
            rz_func,
            transforms.Lambda(partial(augment_with_opt, opt=opt)),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    )
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return path


# ----------------------------
# 🆕 ENHANCED Data augmentation
# ----------------------------

def data_augment(img, opt):
    """
    Enhanced augmentation to prevent overfitting
    """
    img = np.array(img)

    # 🆕 ALWAYS apply if data_aug is enabled (not probabilistic)
    if opt.data_aug:
        # Convert to PIL for easier manipulation
        img_pil = Image.fromarray(img)

        # 🆕 1. Color Jittering (50% chance)
        if random() < 0.5:
            # Brightness
            brightness_factor = uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(brightness_factor)

            # Contrast
            contrast_factor = uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(contrast_factor)

            # Saturation
            saturation_factor = uniform(0.8, 1.2)
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(saturation_factor)

        # Convert back to numpy
        img = np.array(img_pil)

        # 🆕 2. Gaussian Noise (30% chance)
        if random() < 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = np.clip(img.astype(np.int16) + noise,
                          0, 255).astype(np.uint8)

    # 🆕 3. Blur (increased probability)
    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    # 🆕 4. JPEG compression (increased probability)
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
    _, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
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
