import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import argparse
from tqdm import tqdm
import pywt
from options.tSNE_options import tSNE_Options
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class SimpleImageDataset(Dataset):
    """Dataset for loading images from categorized folders with per-category limit."""

    def __init__(self, root_dir, transform=None, wavelet='haar', level=3, max_per_class=500):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.wavelet = wavelet

        print("Wavelet used is ", self.wavelet)
        self.level = level
        self.max_per_class = max_per_class

        # Categories
        self.categories = ['ADM', 'DDIM', 'DDPM', 'DiffSwap', 'GAN', 'Real']

        # Collect all image paths and labels
        self.samples = []
        self.labels = []

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

        for idx, category in enumerate(self.categories):
            category_path = self.root_dir / category
            if not category_path.exists():
                print(f"Warning: Category folder {category} not found!")
                continue

            # Collect all images in this category
            images = []
            for ext in image_extensions:
                images.extend(category_path.glob(ext))

            total = len(images)
            print(f"Category {category} contains {total} images")

            # Limit to max_per_class
            if total > self.max_per_class:
                print(f"Limiting {category} to {self.max_per_class} images...")
                images = np.random.choice(
                    images, self.max_per_class, replace=False)
            else:
                print(f"Using all {total} images in {category}")

            # Store
            for img_path in images:
                self.samples.append(img_path)
                self.labels.append(idx)

        print(f"\nFinal dataset size: {len(self.samples)} images")
        for idx, cat in enumerate(self.categories):
            count = sum(1 for l in self.labels if l == idx)
            print(f"  {cat}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Compute wavelet
        wavelet_packets = self.compute_wavelet_packets(image)

        return image, wavelet_packets, label

    def compute_wavelet_packets(self, image_tensor):
        """
        Use the same wavelet computation as WolterWaveletPacketTrainer.
        """
        import numpy as np
        from networks.FrequencyModels.WaveletsPacketsScratch_128.Trainer_WaveletsPacketsScratch_128 import compute_wavelet_packet_coeffs, log_scale_packets

        # Convert tensor to numpy (H, W, 3)
        img_np = image_tensor.cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        # Compute wavelet packets

        packets = compute_wavelet_packet_coeffs(
            img_np,
            wavelet=self.wavelet,
            level=self.level,
            mode='reflect'  # or self.wavelet_mode
        )

        # Apply log scaling as in training
        packets = log_scale_packets(packets)

        # ⚠️ Only convert if not already a tensor
        if not isinstance(packets, torch.Tensor):
            packets = torch.from_numpy(packets).float()
        else:
            packets = packets.float()

        return packets


def extract_all_embeddings(model, dataloader, device):
    """
    Extract embeddings from RGB, Wavelet, and MHA fusion models.

    Args:
        model: MHAFusionTrainer instance
        dataloader: DataLoader for the dataset
        device: torch device

    Returns:
        rgb_embeddings: numpy array of shape [N, 128]
        wavelet_embeddings: numpy array of shape [N, 128]
        fusion_embeddings: numpy array of shape [N, 128]
        labels: numpy array of shape [N]
    """
    model.eval()
    model.rgb_model.eval()
    model.wavelet_model.eval()
    model.model.eval()

    all_rgb_embeddings = []
    all_wavelet_embeddings = []
    all_fusion_embeddings = []
    all_labels = []

    with torch.no_grad():
        for rgb_imgs, wavelet_imgs, labels in tqdm(dataloader, desc="Extracting embeddings"):
            rgb_imgs = rgb_imgs.to(device)
            wavelet_imgs = wavelet_imgs.to(device)

            # Extract embeddings from base models
            rgb_embed = model.rgb_model(rgb_imgs)  # [B, 128]
            wavelet_embed = model.wavelet_model(wavelet_imgs)  # [B, 128]

            # Get fused embeddings (before final classifier)
            if model.model.fusion_type == 'cross_attention':
                rgb_attended = model.model.rgb_to_wavelet_attn(
                    rgb_embed, wavelet_embed)
                wavelet_attended = model.model.wavelet_to_rgb_attn(
                    wavelet_embed, rgb_embed)
                fused = torch.cat([rgb_attended, wavelet_attended], dim=1)
                fusion_embed = model.model.fusion_layer(fused)

            elif model.model.fusion_type == 'self_attention':
                concat = torch.cat([rgb_embed, wavelet_embed], dim=1)
                x = model.model.input_projection(concat)
                x = x.unsqueeze(1)
                attn_output, _ = model.model.self_attn(x, x, x)
                attn_output = attn_output.squeeze(1)
                x = x.squeeze(1)
                x = model.model.norm(x + attn_output)
                fusion_embed = x + model.model.ffn(x)

            else:  # concat
                concat = torch.cat([rgb_embed, wavelet_embed], dim=1)
                fusion_embed = model.model.fusion_layer(concat)

            # Store all embeddings
            all_rgb_embeddings.append(rgb_embed.cpu().numpy())
            all_wavelet_embeddings.append(wavelet_embed.cpu().numpy())
            all_fusion_embeddings.append(fusion_embed.cpu().numpy())
            all_labels.append(labels.numpy())

    rgb_embeddings = np.vstack(all_rgb_embeddings)
    wavelet_embeddings = np.vstack(all_wavelet_embeddings)
    fusion_embeddings = np.vstack(all_fusion_embeddings)
    labels = np.concatenate(all_labels)

    return rgb_embeddings, wavelet_embeddings, fusion_embeddings, labels


def plot_tsne_comparison(rgb_embeddings, wavelet_embeddings, fusion_embeddings,
                         labels, category_names, output_dir='tsne_outputs',
                         perplexity=30, max_iter=1000):
    """
    Create and save three separate t-SNE visualizations (RGB, Wavelet, Fusion).

    Args:
        rgb_embeddings: numpy array [N, 128]
        wavelet_embeddings: numpy array [N, 128]
        fusion_embeddings: numpy array [N, 128]
        labels: numpy array [N]
        category_names: list of category names
        output_dir: directory to save plots
        perplexity: t-SNE perplexity parameter
        n_iter: number of iterations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define colors for each category
    colors = ["#FF0000", "#FF6600", "#FFFB00", "#FF00E6", "#0008FF", '#98D8C8']

    # Process each embedding type
    embeddings_dict = {
        'RGB Model': rgb_embeddings,
        'Wavelet Model': wavelet_embeddings,
        'MHA Fusion Model': fusion_embeddings
    }

    for model_name, embeddings in embeddings_dict.items():
        print(
            f"\nComputing t-SNE for {model_name} with perplexity={perplexity}...")

        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter,
                    random_state=42, verbose=1, n_jobs=1)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create plot
        plt.figure(figsize=(12, 10))

        # Plot each category
        for idx, category in enumerate(category_names):
            mask = labels == idx
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=colors[idx], label=category, alpha=0.6, s=50,
                        edgecolors='w', linewidth=0.5)

        plt.legend(fontsize=12, markerscale=1.5, loc='best')
        plt.title(f't-SNE Visualization of {model_name} Embeddings (128D)',
                  fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot with appropriate filename
        safe_name = model_name.lower().replace(' ', '_')
        save_path = os.path.join(output_dir, f'tsne_{safe_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

        plt.close()

    # Create a combined comparison plot
    print(f"\nCreating combined comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        print(f"Computing t-SNE for {model_name} (comparison view)...")

        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter,
                    random_state=42, verbose=0)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot on subplot
        ax = axes[ax_idx]
        for idx, category in enumerate(category_names):
            mask = labels == idx
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=colors[idx], label=category, alpha=0.6, s=30,
                       edgecolors='w', linewidth=0.5)

        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=11)
        ax.set_ylabel('t-SNE Component 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, markerscale=1.2)

    plt.suptitle('t-SNE Comparison: RGB vs Wavelet vs MHA Fusion (128D Embeddings)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save combined plot
    combined_path = os.path.join(output_dir, 'tsne_comparison_all.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined comparison: {combined_path}")

    plt.close()

    print(f"\n✓ All t-SNE visualizations saved to: {output_dir}/")
    print(f"  - tsne_rgb_model.png")
    print(f"  - tsne_wavelet_model.png")
    print(f"  - tsne_mha_fusion_model.png")
    print(f"  - tsne_comparison_all.png (combined view)")


def main():
    opt = tSNE_Options().parse()

    # CRITICAL: Set isTrain=True to prevent automatic checkpoint loading in __init__
    opt.isTrain = True
    opt.continue_train = False
    opt.init_gain = 0.02

    # Add missing attributes that MHAFusionTrainer expects
    if not hasattr(opt, 'optim'):
        opt.optim = 'adam'
    if not hasattr(opt, 'lr'):
        opt.lr = 0.0001
    if not hasattr(opt, 'beta1'):
        opt.beta1 = 0.9

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    print("\nLoading dataset...")
    dataset = SimpleImageDataset(
        root_dir=opt.data_root,
        transform=transform,
        wavelet='haar',
        level=opt.wavelet_level
    )

    # Limit number of samples if specified
    if opt.num_samples is not None and opt.num_samples < len(dataset):
        indices = np.random.choice(
            len(dataset), opt.num_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using {opt.num_samples} random samples")

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    print("\nLoading model...")
    from networks.MHA_128.Trainer_MHA_128 import MHAFusionTrainer

    # Initialize model (with isTrain=True, it won't auto-load checkpoint)
    model = MHAFusionTrainer(opt)

    # Now manually load the fusion model checkpoint
    if opt.checkpoint_path and os.path.exists(opt.checkpoint_path):
        print(
            f"\nManually loading MHA Fusion checkpoint from: {opt.checkpoint_path}")
        try:
            checkpoint = torch.load(
                opt.checkpoint_path, map_location=device, weights_only=False)

            if 'model' in checkpoint:
                try:
                    model.model.load_state_dict(
                        checkpoint['model'], strict=True)
                    print("✓ Successfully loaded MHA Fusion model checkpoint")
                except RuntimeError as e:
                    print(f"Warning: Could not load with strict=True")
                    print(f"Error: {e}")
                    print("\nAttempting to load with strict=False...")
                    model.model.load_state_dict(
                        checkpoint['model'], strict=False)
                    print("✓ Loaded checkpoint (some weights may not match)")
            else:
                print("Warning: Checkpoint does not contain 'model' key")
                print("Available keys:", checkpoint.keys())
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with initialized fusion model...")
    else:
        print(f"\nNo valid checkpoint path provided or file not found")
        print("Using initialized fusion model (RGB and Wavelet are pre-trained)")

    # Set models to eval mode
    model.eval()
    model.rgb_model.eval()
    model.wavelet_model.eval()
    model.model.eval()

    print("Model setup complete")

    # Extract embeddings
    print("\nExtracting embeddings from all models...")
    rgb_embeddings, wavelet_embeddings, fusion_embeddings, labels = extract_all_embeddings(
        model, dataloader, device
    )

    print(f"\nExtracted embeddings:")
    print(f"  RGB embeddings shape: {rgb_embeddings.shape}")
    print(f"  Wavelet embeddings shape: {wavelet_embeddings.shape}")
    print(f"  Fusion embeddings shape: {fusion_embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Category names
    category_names = ['ADM', 'DDIM', 'DDPM', 'DiffSwap', 'GAN', 'Real']
    MAX_TSNE_SAMPLES = 10000

    if labels.shape[0] > MAX_TSNE_SAMPLES:
        print(
            f"\nSubsampling dataset to {MAX_TSNE_SAMPLES} samples for t-SNE...")
        idx = np.random.choice(
            labels.shape[0], MAX_TSNE_SAMPLES, replace=False)

        rgb_embeddings = rgb_embeddings[idx]
        wavelet_embeddings = wavelet_embeddings[idx]
        fusion_embeddings = fusion_embeddings[idx]
        labels = labels[idx]
    # Generate t-SNE plots for all three models
    plot_tsne_comparison(
        rgb_embeddings,
        wavelet_embeddings,
        fusion_embeddings,
        labels,
        category_names,
        output_dir=opt.output_dir,
        perplexity=opt.perplexity,
        max_iter=opt.n_iter
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
