import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from validate import validate, validate_wavelet
from data import create_dataloader
from earlystop import EarlyStopping
from networks.FrequencyModels.WaveletsResNet_512_Trainer import Wavelet_ResNet_Trainer
from options.train_options import TrainOptions


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    opt.dataroot = f"{opt.dataroot}/{opt.train_split}/"
    val_opt = get_val_opt()
    # === Data loader ===
    data_loader = create_dataloader(opt)
    print(f"# Training images = {len(data_loader)}")

    # === Logging ===
    train_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "val"))

    # === Model ===
    model = Wavelet_ResNet_Trainer(opt)
    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

    print(f"🚀 Starting Wavelet-ResNet50 training for {opt.niter} epochs...\n")

    # === Validation selector ===
    def evaluate_model(epoch):
        model.eval()
        try:
            # If model is Wavelet_ResNet_Trainer, use wavelet validation helper.
            # We check type to be explicit — adjust if you use a different class name.

            acc_ap = validate_wavelet(model.model, val_opt)

            if acc_ap is None:
                print("⚠️ Validation returned None.")
                model.train()
                return None

            acc, ap = acc_ap[:2]
        except Exception as e:
            print(f"⚠️ Validation failed at epoch {epoch}: {e}")
            model.train()
            return None

        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(f"(Val @ epoch {epoch}) acc: {acc:.6f}; ap: {ap:.6f}")
        model.train()
        return acc

    # === Training loop ===
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{opt.niter} ---")

        model.train()
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            try:
                model.set_input(data)
            except Exception as e:
                print(f"[ERROR] Failed at step {model.total_steps}: {e}")
                sys.exit(1)

            # === Train Step ===
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                try:
                    loss_val = float(model.loss)
                except Exception:
                    loss_val = model.loss
                print(
                    f"Train loss: {loss_val:.6f} at step {model.total_steps}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)
            if model.total_steps % opt.save_latest_freq == 0:
                print(f"💾 Saving latest model (step {model.total_steps})")
                model.save_networks('latest')
                evaluate_model(epoch)

        # === Save epoch checkpoint & validation ===
        print(f"💾 Saving checkpoint at end of epoch {epoch}")
        model.save_networks('latest')
        model.save_networks(epoch)
        acc = evaluate_model(epoch)

        early_stopping(acc, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered — ending training.")
            break

        print(
            f"⏱️ Epoch {epoch + 1} finished in {(time.time() - epoch_start_time):.2f}s")

    print("\n✅ Training completed successfully.")
    train_writer.close()
    val_writer.close()
