import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder, DualInputDataset


# ===========================
# Standard DataLoader (RGB or Wavelet only)
# ===========================

def get_dataset(opt):
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

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
        num_workers=int(opt.num_threads),
        pin_memory=True
    )
    return data_loader


# ===========================
# MHA Fusion DataLoader (RGB + Wavelet)
# ===========================

def get_mha_dataset(opt):
    """
    Create dataset that returns both RGB and Wavelet inputs.
    """
    dset_lst = []
    for cls in opt.classes:
        root = opt.dataroot + '/' + cls
        # Use DualInputDataset instead of standard dataset
        dset = DualInputDataset(opt, root)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


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
