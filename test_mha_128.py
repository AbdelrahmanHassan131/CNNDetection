import os
import csv
import torch
import numpy as np
import pywt
from PIL import Image
import torchvision.transforms as transforms

from validate_mha import validate_mha
from networks.MHA_128.Trainer_MHA_128 import MHAFusionTrainer
from options.test_options import TestOptions


# ===========================
# Wavelet Processing Functions
# ===========================

def compute_wavelet_packet_coeffs(img, wavelet='haar', level=3, mode='reflect'):
    """Compute wavelet packet coefficients for an RGB image."""
    if torch.is_tensor(img):
        img = img.cpu().numpy()

    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    H, W, _ = img.shape
    all_packets = []

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

    for c in range(3):
        channel = img[:, :, c]
        wp = pywt.WaveletPacket2D(
            data=channel, wavelet=wavelet, mode=mode, maxlevel=level)

        for path in packet_paths:
            coeff = wp[path].data
            all_packets.append(coeff)

    all_packets = np.array(all_packets, dtype=np.float32)
    return all_packets


def log_scale_packets(packets, epsilon=1e-10):
    """Apply log-scaling to packet coefficients."""
    return np.sign(packets) * np.log(np.abs(packets) + epsilon)


def prepare_rgb_input(img, transform):
    """Prepare RGB input for Wang2020 model."""
    return transform(img)


def prepare_wavelet_input(img, wavelet_type='haar', level=3, mode='reflect', use_log=True):
    """Prepare wavelet input for Wolter2022 model."""
    img_array = np.array(img)
    wavelet_coeffs = compute_wavelet_packet_coeffs(img_array, wavelet=wavelet_type,
                                                   level=level, mode=mode)
    if use_log:
        wavelet_coeffs = log_scale_packets(wavelet_coeffs)

    return torch.from_numpy(wavelet_coeffs).float()


# ===========================
# Testing Functions
# ===========================

def test_single_image(model, image_path, opt):
    """Test a single image and return prediction."""
    img = Image.open(image_path).convert('RGB')

    # RGB transform
    if opt.no_resize:
        rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        rgb_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # Prepare inputs
    rgb_tensor = prepare_rgb_input(img, rgb_transform).unsqueeze(0).cuda()
    wavelet_tensor = prepare_wavelet_input(
        img,
        wavelet_type=opt.wavelet_type,
        level=opt.wavelet_level,
        mode='reflect',
        use_log=True
    ).unsqueeze(0).cuda()

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(rgb_tensor, wavelet_tensor).sigmoid().item()

    prediction = "FAKE" if output > 0.5 else "REAL"
    confidence = output if output > 0.5 else (1 - output)

    return prediction, confidence, output


def test_dataset(model, opt, dataroot, dataset_name):
    """Test model on a full dataset."""
    opt.dataroot = dataroot
    opt.classes = os.listdir(opt.dataroot) if os.path.isdir(
        opt.dataroot) else ['']

    acc, ap, r_acc, f_acc, _, _ = validate_mha(model, opt)

    print(f"\n{'='*60}")
    print(f"Results on {dataset_name}:")
    print(f"  Overall Accuracy:    {acc:.4f}")
    print(f"  Average Precision:   {ap:.4f}")
    print(f"  Real Images Acc:     {r_acc:.4f}")
    print(f"  Fake Images Acc:     {f_acc:.4f}")
    print(f"{'='*60}")

    return acc, ap, r_acc, f_acc


# ===========================
# Main Testing Script
# ===========================

if __name__ == '__main__':
    # Parse options
    opt = TestOptions().parse(print_options=False)

    print(f"\n[DEBUG] After parsing:")
    print(f"  opt.dataroot: {opt.dataroot}")
    print(f"  opt.classes: {opt.classes}")
    print()

    # Set testing parameters
    opt.isTrain = False
    opt.serial_batches = True
    opt.jpg_method = ['pil']
    opt.compute_wavelets = True
    opt.wavelet_mode = 'reflect'
    opt.use_log_packets = True

    # Determine test mode
    if opt.image_path is not None:
        test_mode = 'single'
    elif len(opt.test_splits) > 1:
        test_mode = 'batch'
    else:
        test_mode = 'dataset'

    # Create results directory
    os.makedirs(opt.results_dir, exist_ok=True)

    # Load model
    print("\n" + "="*80)
    print("🔬 LOADING MHA FUSION MODEL")
    print("="*80)
    print(f"📦 Model checkpoint: {opt.model_path}")
    print(f"🖼️  RGB base model:   {opt.rgb_model_path}")
    print(f"🌊 Wavelet base model: {opt.wavelet_model_path}")
    print(f"🌊 Wavelet: {opt.wavelet_type}, level: {opt.wavelet_level}")
    print("="*80 + "\n")

    # Override the checkpoints_dir and name to point to the correct model path
    # This prevents MHAFusionTrainer from trying to load from default location
    opt.checkpoints_dir = os.path.dirname(os.path.dirname(opt.model_path))
    opt.name = os.path.basename(os.path.dirname(opt.model_path))
    opt.epoch = os.path.basename(opt.model_path).replace(
        'model_epoch_', '').replace('.pth', '')

    model = MHAFusionTrainer(opt)
    print(f"✅ Model loaded successfully\n")

    model.cuda()
    model.eval()

    # Run testing
    if test_mode == 'single':
        # Test single image
        if not os.path.exists(opt.image_path):
            print(f"❌ Error: Image not found at {opt.image_path}")
            exit(1)

        print(f"🔍 Testing image: {opt.image_path}")
        prediction, confidence, raw_score = test_single_image(
            model, opt.image_path, opt)

        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Image:       {os.path.basename(opt.image_path)}")
        print(f"Prediction:  {prediction}")
        print(f"Confidence:  {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"Raw Score:   {raw_score:.4f}")
        print(f"  (Score > 0.5 = REAL, Score < 0.5 = FAKE)")
        print("="*60)

    elif test_mode == 'dataset':
        # Test on single dataset
        test_path = f"{opt.dataroot}/{opt.test_splits[0]}"
        print(f"🧪 Testing on dataset: {test_path}\n")
        acc, ap, r_acc, f_acc = test_dataset(
            model, opt, test_path, opt.test_splits[0])

        # Save results
        model_name = os.path.basename(opt.model_path).replace('.pth', '')
        csv_name = os.path.join(opt.results_dir, f'{model_name}_results.csv')

        with open(csv_name, 'w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow([f"{model_name} - MHA Fusion Model Testing"])
            csv_writer.writerow(
                ['Dataset', 'Accuracy', 'Avg Precision', 'Real Acc', 'Fake Acc'])
            csv_writer.writerow([opt.test_splits[0], f"{acc:.4f}", f"{ap:.4f}",
                                f"{r_acc:.4f}", f"{f_acc:.4f}"])

        print(f"\n✅ Results saved to: {csv_name}")

    elif test_mode == 'batch':
        # Test on multiple datasets
        model_name = os.path.basename(opt.model_path).replace('.pth', '')
        rows = [[f"{model_name} - MHA Fusion Model Testing"],
                ['Dataset', 'Accuracy', 'Avg Precision', 'Real Acc', 'Fake Acc']]

        print(f"🧪 Testing {model_name} on multiple datasets...\n")

        for test_split in opt.test_splits:
            test_path = f"{opt.dataroot}/{test_split}"

            if not os.path.exists(test_path):
                print(f"⚠️  Skipping {test_split}: Path not found")
                continue

            acc, ap, r_acc, f_acc = test_dataset(
                model, opt, test_path, test_split)
            rows.append([test_split, f"{acc:.4f}", f"{ap:.4f}",
                        f"{r_acc:.4f}", f"{f_acc:.4f}"])

        # Save batch results
        csv_name = os.path.join(
            opt.results_dir, f'{model_name}_batch_results.csv')
        with open(csv_name, 'w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerows(rows)

        print(f"\n✅ Batch results saved to: {csv_name}")

    print("\n" + "="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)
