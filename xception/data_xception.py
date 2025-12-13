"""
Simple data loader for Xception model evaluation
Only handles basic image loading and resizing - no augmentation
"""

import torch
from torchvision import datasets, transforms
from PIL import Image


def create_xception_dataloader(dataroot, batch_size=32, num_workers=0, shuffle=False):
    """
    Create a simple dataloader for Xception evaluation.
    
    Args:
        dataroot: Path to dataset with 'fake' and 'real' subfolders
        batch_size: Batch size for evaluation
        num_workers: Number of worker threads (0 for Windows to avoid pickle errors)
        shuffle: Whether to shuffle data (False for evaluation)
    
    Returns:
        DataLoader with (images, labels) where labels are 0=fake, 1=real
    """
    
    # Xception-specific transforms
    # Resize to 333, center crop to 299, normalize with mean/std of 0.5
    transform = transforms.Compose([
        transforms.Resize(333),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset using ImageFolder
    # Expects structure: dataroot/fake/*.jpg, dataroot/real/*.jpg
    # Labels: fake=0, real=1 (alphabetical order)
    dataset = datasets.ImageFolder(
        root=dataroot,
        transform=transform,
        loader=lambda path: Image.open(path).convert('RGB')
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def create_dataloader_xception(opt):
    """
    Create dataloader for Xception using options object (for training script compatibility).
    
    Args:
        opt: Options object with attributes:
            - dataroot: Path to dataset
            - batch_size: Batch size
            - num_threads: Number of workers
            - serial_batches: Whether to use sequential loading (no shuffle if True)
            
    Returns:
        DataLoader with (images, labels)
    """
    shuffle = not opt.serial_batches if hasattr(opt, 'serial_batches') else True
    num_workers = opt.num_threads if hasattr(opt, 'num_threads') else 0
    batch_size = opt.batch_size if hasattr(opt, 'batch_size') else 32
    
    return create_xception_dataloader(
        dataroot=opt.dataroot,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )


def get_dataset_info(dataroot):
    """
    Get information about the dataset.
    
    Args:
        dataroot: Path to dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset = datasets.ImageFolder(root=dataroot)
    
    info = {
        'total_samples': len(dataset),
        'num_classes': len(dataset.classes),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'samples_per_class': {}
    }
    
    # Count samples per class
    for class_name in dataset.classes:
        class_idx = dataset.class_to_idx[class_name]
        count = sum(1 for _, label in dataset.samples if label == class_idx)
        info['samples_per_class'][class_name] = count
    
    return info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Xception dataloader')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to validation dataset (containing fake and real folders)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for loading')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of worker threads (use 0 for Windows)')
    
    args = parser.parse_args()
    
    # Get dataset info
    print("Loading dataset information...")
    info = get_dataset_info(args.dataroot)
    print("\nDataset Information:")
    print(f"  Total samples: {info['total_samples']}")
    print(f"  Classes: {info['classes']}")
    print(f"  Class mapping: {info['class_to_idx']}")
    print(f"  Samples per class:")
    for class_name, count in info['samples_per_class'].items():
        print(f"    {class_name}: {count}")
    
    # Create dataloader
    print(f"\nCreating dataloader...")
    dataloader = create_xception_dataloader(
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Test loading one batch
    print(f"\nTesting batch loading...")
    images, labels = next(iter(dataloader))
    print(f"  Batch image shape: {images.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Unique labels in batch: {labels.unique().tolist()}")
    print("\n✅ Dataloader test successful!")

