import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Xception-specific normalization (as specified in the model documentation)
XCEPTION_MEAN = [0.5, 0.5, 0.5]
XCEPTION_STD = [0.5, 0.5, 0.5]


def get_dataset(opt):
    """Create dataset from directory structure with fake and real folders"""
    # Cropping
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    # Flipping
    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
        
    # Resizing
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Resize(opt.loadSize, interpolation=Image.BILINEAR)

    # Use Xception-specific normalization
    normalize = transforms.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD)

    # Use ImageFolder directly on dataroot
    # ImageFolder will automatically find 'fake' and 'real' folders
    # and assign labels alphabetically: fake=0, real=1
    dset = datasets.ImageFolder(
        opt.dataroot,
        transforms.Compose([
            rz_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            normalize,
        ])
    )
    return dset


def binary_dataset(opt, root):
    """Create binary classification dataset with Xception transforms"""
    
    # Cropping
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    # Flipping
    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
        
    # Resizing
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Resize(opt.loadSize, interpolation=Image.BILINEAR)

    # Use Xception-specific normalization
    normalize = transforms.Normalize(mean=XCEPTION_MEAN, std=XCEPTION_STD)

    dset = datasets.ImageFolder(
        root,
        transforms.Compose([
            rz_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            normalize,
        ])
    )
    return dset


def get_bal_sampler(dataset):
    """Create balanced sampler for imbalanced datasets"""
    # Get targets directly from the dataset (ImageFolder has .targets attribute)
    targets = dataset.targets

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights)
    )
    return sampler


def create_dataloader(opt):
    """Create dataloader"""
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0  # Set to 0 to avoid pickle errors with lambda functions on Windows
    )
    return data_loader
