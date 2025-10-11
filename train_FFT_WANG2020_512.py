import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.FrequencyModels.FFT_512_Trainer import FFT_512_Trainer
from options.train_options import TrainOptions


def get_val_opt():
    """Create validation options identical to your Wang2020 setup."""
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = f"{val_opt.dataroot}/{val_opt.val_split}/"
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']

    # Simplify blur and jpeg parameters
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.dataroot = f"{opt.dataroot}/{opt.train_split}/"
    val_opt = get_val_opt()

    # === DATA LOADER ===
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print(f"#training images = {dataset_size}")

    # === LOGGING ===
    train_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "val"))

    # === MODEL ===
    model = FFT_512_Trainer(opt)
    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0
        print(f"\n--- Epoch {epoch+1}/{opt.niter} ---")

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            try:
                model.set_input(data)
            except Exception as e:
                print(
                    f"[ERROR] Failed to load image or label at step {model.total_steps}: {e}")
                print("⛔ Stopping training. Please fix the problematic data.")
                sys.exit(1)

            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(
                    f"Train loss: {model.loss:.6f} at step: {model.total_steps}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print(
                    f"Saving latest model (epoch {epoch}, step {model.total_steps})")
                model.save_networks('latest')

        # Save model at epoch end
        if epoch % opt.save_epoch_freq == 0:
            print(
                f"Saving model checkpoint at end of epoch {epoch}, step {model.total_steps}")
            model.save_networks('latest')
            model.save_networks(epoch)

        # === VALIDATION ===
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continuing training...")
                early_stopping = EarlyStopping(
                    patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping triggered — ending training.")
                break

        model.train()
