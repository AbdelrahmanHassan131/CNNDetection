import os
import sys
import time
import torch
import torch.nn as nn
import argparse
from torch.utils.tensorboard import SummaryWriter

from validate_fusion import validate_fusion
from data import create_mha_dataloader
from earlystop import EarlyStopping
from networks.Fusion_128.Trainer_Fusion_128 import ConcatenationFusionTrainer
from options.train_options import TrainOptions


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = f"{val_opt.dataroot}/{val_opt.val_split}/"
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']

    # Enable both RGB and wavelet computation
    val_opt.compute_wavelets = True

    # Handle blur/jpg options properly
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == "__main__":
    opt = TrainOptions().parse()
    opt.dataroot = f"{opt.dataroot}/{opt.train_split}/"

    # Enable wavelet computation for dual input
    opt.compute_wavelets = True

    val_opt = get_val_opt()

    print("=" * 80)
    print("🔗 CONCATENATION FUSION TRAINING")
    print("=" * 80)
    print(f"📊 Dataset: {opt.dataroot}")
    print(f"🖼️  RGB Model: {opt.rgb_model_path}")
    print(f"🌊 Wavelet Model: {opt.wavelet_model_path}")
    print(f"🔀 Fusion Type: Simple Concatenation + MLP")
    print(
        f"❄️  Base Models Frozen: {getattr(opt, 'freeze_base_models', True)}")
    print(f"👷 DataLoader workers: {opt.num_threads}")
    print(f"🎯 Batch size: {opt.batch_size}")
    print("=" * 80)

    # Create dataloader with dual input (RGB + Wavelet)
    data_loader = create_mha_dataloader(opt)
    dataset_size = len(data_loader)
    print(f"# training images = {dataset_size * opt.batch_size}")
    print(f"# training batches = {dataset_size}")

    # Create TensorBoard log directories
    train_logdir = os.path.join(opt.checkpoints_dir, opt.name, "train")
    val_logdir = os.path.join(opt.checkpoints_dir, opt.name, "val")
    os.makedirs(train_logdir, exist_ok=True)
    os.makedirs(val_logdir, exist_ok=True)

    train_writer = SummaryWriter(train_logdir)
    val_writer = SummaryWriter(val_logdir)

    # Create fusion model
    model = ConcatenationFusionTrainer(opt)

    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True
    )

    print("\n🚀 Starting training...")
    print("-" * 80)

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            # Data contains (rgb_images, wavelet_packets, labels)
            model.set_input(data)
            model.optimize_parameters()

            # Log training loss
            if model.total_steps % opt.loss_freq == 0:
                print(
                    f"Train loss: {model.loss:.6f} at step: {model.total_steps}")
                train_writer.add_scalar("loss", model.loss, model.total_steps)

            # Save and validate periodically
            if model.total_steps % opt.save_latest_freq == 0:
                print(
                    f"Saving the latest model {opt.name} "
                    f"(epoch {epoch}, total steps {model.total_steps})"
                )
                model.save_networks("latest")

        # End of epoch save
        if epoch % opt.save_epoch_freq == 0:
            print(
                f"Saving model at end of epoch {epoch}, iters {model.total_steps}")
            model.save_networks("latest")
            model.save_networks(epoch)

        # Validation after each epoch
        model.eval()
        acc, ap = validate_fusion(model, val_opt)[:2]
        val_writer.add_scalar("accuracy", acc, model.total_steps)
        val_writer.add_scalar("ap", ap, model.total_steps)

        epoch_time = time.time() - epoch_start_time
        print(
            f"(Val @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f} | Time: {epoch_time:.2f}s"
        )

        # Early stopping logic
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(
                    patience=opt.earlystop_epoch, delta=-0.002, verbose=True
                )
            else:
                print("Early stopping triggered. Training complete.")
                break

        model.train()

    # Close the writers cleanly
    train_writer.close()
    val_writer.close()

    print("\n" + "=" * 80)
    print("✅ Training finished successfully!")
    print("=" * 80)
    print(f"📁 TensorBoard logs saved in:")
    print(f"   - Training: {train_logdir}")
    print(f"   - Validation: {val_logdir}")
    print(f"\n📊 View logs with:")
    print(
        f"   tensorboard --logdir {os.path.join(opt.checkpoints_dir, opt.name)}")
    print("=" * 80)
