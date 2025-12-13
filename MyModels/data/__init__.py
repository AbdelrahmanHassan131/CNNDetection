import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder, DualInputDataset


# ===========================
# Standard DataLoader (RGB or Wavelet only)
# ===========================

def has_subfolders(directory):
    """Check if directory contains subfolders (training structure) or direct images (validation structure)"""
    import os
    if not os.path.exists(directory):
        return False
    items = os.listdir(directory)
    # Check if any item is a directory
    for item in items:
        if os.path.isdir(os.path.join(directory, item)):
            return True
    return False


def get_dataset(opt):
    """Smart dataloader - handles both training structure (subfolders) and validation structure (direct images)"""
    from .datasets import dataset_folder
    
    # Check if we have the training structure (fake/method1/, fake/method2/) 
    # or validation structure (fake/img1.jpg, fake/img2.jpg)
    first_class_path = opt.dataroot + '/' + opt.classes[0]
    
    if has_subfolders(first_class_path):
        # Training structure: fake/method1/, real/dataset1/, etc.
        print("Detected training structure (subfolders)")
        dset_lst = []
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    else:
        # Validation structure: fake/img.jpg, real/img.jpg
        print("Detected validation structure (direct images)")
        # Use dataroot directly - ImageFolder will find fake and real folders
        dset = dataset_folder(opt, opt.dataroot)
        return dset


def get_bal_sampler(dataset):
    """Balanced sampler - handles both ConcatDataset and single ImageFolder"""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # Training structure - ConcatDataset
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
    else:
        # Validation structure - single ImageFolder
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
    """Standard dataloader for single-input models (RGB or Wavelet)"""
    shuffle = not opt.serial_batches if (
        opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,  # Set to 0 to avoid pickle errors on Windows
        pin_memory=True
    )
    return data_loader


# ===========================
# MHA Fusion DataLoader (RGB + Wavelet)
# ===========================

def get_mha_dataset(opt):
    """
    Create dataset that returns both RGB and Wavelet inputs.
    Smart detection for training vs validation folder structure.
    """
    from .datasets import DualInputDataset
    
    # Check if we have training structure (subfolders) or validation structure (direct images)
    first_class_path = opt.dataroot + '/' + opt.classes[0]
    
    if has_subfolders(first_class_path):
        # Training structure: fake/method1/, real/dataset1/, etc.
        print("Detected training structure (subfolders) for MHA")
        dset_lst = []
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = DualInputDataset(opt, root)
            dset_lst.append(dset)
        return torch.utils.data.ConcatDataset(dset_lst)
    else:
        # Validation structure: fake/img.jpg, real/img.jpg
        print("Detected validation structure (direct images) for MHA")
        # Use dataroot directly - DualInputDataset will find fake and real folders
        dset = DualInputDataset(opt, opt.dataroot)
        return dset


def create_mha_dataloader(opt):
    """
    DataLoader for MHA Fusion model.
    Returns: (rgb_images, wavelet_packets, labels)
    """
    shuffle = not opt.serial_batches if (
        opt.isTrain and not opt.class_bal) else False
    dataset = get_mha_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(opt.num_threads),
        pin_memory=True
    )
    return data_loader
