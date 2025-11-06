import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from torch.utils.tensorboard import SummaryWriter  # ✅ use built-in PyTorch version

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.FrequencyModels.WaveletsPacketsScratch_128.Trainer_WaveletsPacketsScratch_128 import WolterWaveletPacketTrainer
from options.train_options import TrainOptions


"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = f"{val_opt.dataroot}/{val_opt.val_split}/"
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']

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
    val_opt = get_val_opt()

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print(f"# training images = {dataset_size}")

    # ✅ create TensorBoard log directories (train & val)
    train_logdir = os.path.join(opt.checkpoints_dir, opt.name, "train")
    val_logdir = os.path.join(opt.checkpoints_dir, opt.name, "val")
    os.makedirs(train_logdir, exist_ok=True)
    os.makedirs(val_logdir, exist_ok=True)

    train_writer = SummaryWriter(train_logdir)
    val_writer = SummaryWriter(val_logdir)

    model = WolterWaveletPacketTrainer(opt)
    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True
    )

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

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
                # ✅ Pass trainer (now callable)
                acc, ap = validate(model, val_opt)[:2]
                val_writer.add_scalar("accuracy", acc, model.total_steps)
                val_writer.add_scalar("ap", ap, model.total_steps)
                print(f"(Val @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f}")
                model.train()  # ✅ Now this works

        # ✅ End of epoch save
        if epoch % opt.save_epoch_freq == 0:
            print(
                f"Saving model at end of epoch {epoch}, iters {model.total_steps}")
            model.save_networks("latest")
            model.save_networks(epoch)

            # ✅ Validation after each epoch
            model.eval()
            # ✅ Pass trainer (now callable)
            acc, ap = validate(model, val_opt)[:2]
            val_writer.add_scalar("accuracy", acc, model.total_steps)
            val_writer.add_scalar("ap", ap, model.total_steps)
            print(f"(Val @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f}")
            model.train()  # ✅ Now this works

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

    print(f"\n✅ Training finished. TensorBoard logs saved in:")
    print(f"   - {train_logdir}")
    print(f"   - {val_logdir}")
    print("You can now open TensorBoard anytime with:")
    print(
        f"   tensorboard --logdir {os.path.join(opt.checkpoints_dir, opt.name)}")
