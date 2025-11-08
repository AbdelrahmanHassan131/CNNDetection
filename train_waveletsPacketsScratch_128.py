import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.FrequencyModels.WaveletsPacketsScratch_128.Trainer_WaveletsPacketsScratch_128 import WolterWaveletPacketTrainer
from options.train_options import TrainOptions


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = f"{val_opt.dataroot}/{val_opt.val_split}/"
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']

    # ✅ Ensure wavelet computation is enabled for validation
    val_opt.compute_wavelets = True

    # handle blur/jpg options properly
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

    # ✅ Enable wavelet computation in dataloader
    opt.compute_wavelets = True

    val_opt = get_val_opt()

    print("=" * 80)
    print("🔬 WAVELET-PACKET DEEPFAKE DETECTION TRAINING")
    print("=" * 80)
    print(f"📊 Dataset: {opt.dataroot}")
    print(
        f"🌊 Wavelet: {getattr(opt, 'wavelet_type', 'haar')}, Level: {getattr(opt, 'wavelet_level', 3)}")
    print(f"💾 Log scaling: {getattr(opt, 'use_log_packets', True)}")
    print(
        f"👷 DataLoader workers: {opt.num_threads} (parallel wavelet computation)")
    print(f"🎯 Batch size: {opt.batch_size}")
    print("=" * 80)

    # Create dataloader with parallel wavelet computation
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print(f"# training images = {dataset_size * opt.batch_size}")
    print(f"# training batches = {dataset_size}")

    # ✅ create TensorBoard log directories (train & val)
    train_logdir = os.path.join(opt.checkpoints_dir, opt.name, "train")
    val_logdir = os.path.join(opt.checkpoints_dir, opt.name, "val")
    os.makedirs(train_logdir, exist_ok=True)
    os.makedirs(val_logdir, exist_ok=True)

    train_writer = SummaryWriter(train_logdir)
    val_writer = SummaryWriter(val_logdir)

    # Create model
    model = WolterWaveletPacketTrainer(opt)

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

            # ✅ Data is already wavelet packets from dataloader!
            # set_input just transfers to GPU (fast!)
            model.set_input(data)
            model.optimize_parameters()

            # ✅ Log training loss
            if model.total_steps % opt.loss_freq == 0:
                print(
                    f"Train loss: {model.loss:.6f} at step: {model.total_steps}")
                train_writer.add_scalar("loss", model.loss, model.total_steps)

            # ✅ Save and validate periodically
            if model.total_steps % opt.save_latest_freq == 0:
                print(
                    f"Saving the latest model {opt.name} "
                    f"(epoch {epoch}, total steps {model.total_steps})"
                )
                model.save_networks("latest")

                model.eval()
                # ✅ Pass trainer object (callable via __call__)
                acc, ap = validate(model, val_opt)[:2]
                val_writer.add_scalar("accuracy", acc, model.total_steps)
                val_writer.add_scalar("ap", ap, model.total_steps)
                print(f"(Val @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f}")
                model.train()

        # ✅ End of epoch save
        if epoch % opt.save_epoch_freq == 0:
            print(
                f"Saving model at end of epoch {epoch}, iters {model.total_steps}")
            model.save_networks("latest")
            model.save_networks(epoch)

        # ✅ Validation after each epoch
        model.eval()
        acc, ap = validate(model, val_opt)[:2]
        val_writer.add_scalar("accuracy", acc, model.total_steps)
        val_writer.add_scalar("ap", ap, model.total_steps)

        epoch_time = time.time() - epoch_start_time
        print(
            f"(Val @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f} | Time: {epoch_time:.2f}s")

        # ✅ Early stopping logic
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

    # ✅ Close the writers cleanly (important!)
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
