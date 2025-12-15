"""
Clean data loader for Wavelet Packet model
Only handles wavelet packet computation - no augmentation, blur, or JPEG compression
"""

import torch
import numpy as np
import pywt
from torchvision import datasets, transforms
from PIL import Image


def compute_wavelet_packet_coeffs(img, wavelet='haar', level=3, mode='reflect'):
    """
    Compute wavelet packet coefficients for an RGB image.
    
    Args:
        img: numpy array (H, W, 3)
        wavelet: wavelet type (default: 'haar')
        level: decomposition level (default: 3)
        mode: signal extension mode (default: 'reflect')
    
    Returns:
        numpy array of shape (C, H', W') where C = 3 * 4^level
    """
    H, W, _ = img.shape
    all_packets = []
    
    # Generate all wavelet packet paths
    def get_paths(level):
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


class WaveletDataset(datasets.ImageFolder):
    """
    Dataset that computes wavelet packets on-the-fly.
    Only basic preprocessing - no augmentation.
    """
    
    def __init__(self, root, image_size=224, wavelet_type='haar', 
                 wavelet_level=3, wavelet_mode='reflect', use_log_packets=True):
        """
        Args:
            root: Path to dataset (should contain 'fake' and 'real' subfolders)
            image_size: Size to resize images to (default: 224)
            wavelet_type: Wavelet type (default: 'haar')
            wavelet_level: Decomposition level (default: 3)
            wavelet_mode: Signal extension mode (default: 'reflect')
            use_log_packets: Whether to apply log-scaling (default: True)
        """
        self.image_size = image_size
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.wavelet_mode = wavelet_mode
        self.use_log_packets = use_log_packets
        
        # Simple resize and center crop - no augmentation
        self.image_transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # Resize to 256 for 224 crop
            transforms.CenterCrop(image_size),
        ])
        
        # Initialize ImageFolder
        super().__init__(root, transform=None, loader=lambda path: Image.open(path).convert('RGB'))
    
    def __getitem__(self, index):
        """
        Load image and compute wavelet packets.
        
        Returns:
            tuple: (wavelet_tensor, label)
        """
        path, target = self.samples[index]
        
        # Load image
        img = self.loader(path)
        
        # Apply resize and crop
        img = self.image_transform(img)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Compute wavelet packet coefficients
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
        wavelet_tensor = torch.from_numpy(wavelet_coeffs).float()
        
        return wavelet_tensor, target


def create_wavelet_dataloader(dataroot, batch_size=32, num_workers=0, shuffle=False,
                              image_size=224, wavelet_type='haar', wavelet_level=3):
    """
    Create a clean dataloader for Wavelet Packet model.
    
    Args:
        dataroot: Path to dataset with 'fake' and 'real' subfolders
        batch_size: Batch size
        num_workers: Number of worker threads (0 for Windows)
        shuffle: Whether to shuffle data
        image_size: Size to resize images to (default: 224)
        wavelet_type: Wavelet type (default: 'haar')
        wavelet_level: Decomposition level (default: 3)
    
    Returns:
        DataLoader with (wavelet_packets, labels) where labels are 0=fake, 1=real
    """
    dataset = WaveletDataset(
        root=dataroot,
        image_size=image_size,
        wavelet_type=wavelet_type,
        wavelet_level=wavelet_level
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def create_dataloader_wavelet(opt):
    """
    Create dataloader using options object (for training script compatibility).
    
    Args:
        opt: Options object with attributes:
            - dataroot: Path to dataset
            - batch_size: Batch size
            - num_threads: Number of workers
            - serial_batches: Whether to use sequential loading
            - cropSize: Image size (default: 224)
            - wavelet_type: Wavelet type (default: 'haar')
            - wavelet_level: Wavelet level (default: 3)
    
    Returns:
        DataLoader
    """
    shuffle = not opt.serial_batches if hasattr(opt, 'serial_batches') else True
    num_workers = opt.num_threads if hasattr(opt, 'num_threads') else 0
    batch_size = opt.batch_size if hasattr(opt, 'batch_size') else 32
    image_size = opt.cropSize if hasattr(opt, 'cropSize') else 224
    wavelet_type = opt.wavelet_type if hasattr(opt, 'wavelet_type') else 'haar'
    wavelet_level = opt.wavelet_level if hasattr(opt, 'wavelet_level') else 3
    
    return create_wavelet_dataloader(
        dataroot=opt.dataroot,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        image_size=image_size,
        wavelet_type=wavelet_type,
        wavelet_level=wavelet_level
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Wavelet dataloader')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='Path to validation dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    
    args = parser.parse_args()
    
    print("Creating wavelet dataloader...")
    dataloader = create_wavelet_dataloader(
        dataroot=args.dataroot,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Test loading one batch
    print("\nTesting batch loading...")
    wavelet_packets, labels = next(iter(dataloader))
    print(f"  Wavelet packets shape: {wavelet_packets.shape}")  # Should be [B, 192, H/16, W/16]
    print(f"  Labels shape: {labels.shape}")
    print(f"  Wavelet value range: [{wavelet_packets.min():.3f}, {wavelet_packets.max():.3f}]")
    print(f"  Unique labels: {labels.unique().tolist()}")
    print("\n✅ Dataloader test successful!")
