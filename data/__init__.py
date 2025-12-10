import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder


def get_dataset(opt):
    """Create dataset from directory structure with fake and real folders"""
    from .datasets import binary_dataset
    # Use binary_dataset directly on dataroot
    # ImageFolder will automatically find 'fake' and 'real' folders
    # and assign labels alphabetically: fake=0, real=1
    dset = binary_dataset(opt, opt.dataroot)
    return dset


def get_bal_sampler(dataset):
    """Create balanced sampler for imbalanced datasets"""
    # Get targets directly from the dataset (ImageFolder has .targets attribute)
    targets = dataset.targets

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=0)  # Set to 0 to avoid pickle errors on Windows
    return data_loader
