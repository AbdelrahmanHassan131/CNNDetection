import os
import sys
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from validate import validate_wavelet
from data import create_dataloader
from earlystop import EarlyStopping
from networks.FrequencyModels.WaveletsResNet_512_Trainer import Wavelet_ResNet_Trainer
from options.train_options import TrainOptions


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = f"{val_opt.dataroot}/{val_opt.val_split}/"
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ["pil"]

    # normalize blur_sig/jpg_qual
    if hasattr(val_opt, "blur_sig") and isinstance(val_opt.blur_sig, (list, tuple)) and len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if hasattr(val_opt, "jpg_qual") and isinstance(val_opt.jpg_qual, (list, tuple)) and len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt


if __name__ == "__main__":
    # setup deterministic performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # GPU info
    print("=" * 60)
    print("🎮 GPU INFORMATION")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ No CUDA GPUs available!")
    print("=" * 60)
    print()

    opt = TrainOptions().parse()
    opt.dataroot = f"{opt.dataroot}/{opt.train_split}/"
    val_opt = get_val_opt()

    # dataloader
    print("📦 Creating data loader...")
    data_loader = create_dataloader(opt)
    print(f"✓ Training batches per epoch: {len(data_loader)}")
    print(f"✓ Total images per epoch: {len(data_loader) * opt.batch_size}")

    if getattr(opt, "gpu_ids", None) and len(opt.gpu_ids) > 1:
        effective_batch = opt.batch_size * len(opt.gpu_ids)
        print(f"✓ Effective batch size (multi-GPU): {effective_batch}")
        print(f"  ({opt.batch_size} per GPU × {len(opt.gpu_ids)} GPUs)")
    print()

    # loggers
    train_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "val"))

    # model
    model = Wavelet_ResNet_Trainer(opt)
    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=0.001, verbose=True)

    print(f"🚀 Starting Wavelet-ResNet50 training for {opt.niter} epochs...")
    print(f"📋 Initial LR: {opt.lr:.2e}")
    print(f"📋 Batch size per GPU: {opt.batch_size}")
    if getattr(opt, "gpu_ids", None) and len(opt.gpu_ids) > 1:
        print(
            f"📋 Total effective batch size: {opt.batch_size * len(opt.gpu_ids)}")
    print(f"📋 Device: {model.device}")
    print(f"📋 Data augmentation: {'Enabled' if opt.data_aug else 'Disabled'}")
    print(f"📋 Class balancing: {'Enabled' if opt.class_bal else 'Disabled'}")
    print()

    # tracking metrics
    best_acc = 0.0
    best_ap = 0.0
    epoch_losses = []
    epoch_accs = []

    # validation helper
    def evaluate_model(epoch):
        model.eval()
        try:
            net_for_val = model.model.module if hasattr(
                model.model, "module") else model.model
            acc_ap = validate_wavelet(net_for_val, val_opt)
            if acc_ap is None:
                print("⚠️ Validation returned None.")
                model.train()
                return None
            acc, ap = acc_ap[:2]
        except Exception as e:
            print(f"⚠️ Validation failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            model.train()
            return None

        val_writer.add_scalar("accuracy", acc, model.total_steps)
        val_writer.add_scalar("ap", ap, model.total_steps)

        improved = ""
        nonlocal_best = False
        if acc > evaluate_model.best_acc:
            evaluate_model.best_acc = acc
            improved += " 🆙 ACC"
            nonlocal_best = True
        if ap > evaluate_model.best_ap:
            evaluate_model.best_ap = ap
            improved += " 🆙 AP"
            nonlocal_best = True

        print(f"(Val @ epoch {epoch}) acc: {acc:.6f}; ap: {ap:.6f}{improved}")
        print(
            f"              Best so far → acc: {evaluate_model.best_acc:.6f}; ap: {evaluate_model.best_ap:.6f}")

        # save checkpoint if improved
        if nonlocal_best:
            print("💾 Saving best model (improved validation)")
            model.save_networks("best")

        model.train()
        return acc

    # attach static attributes
    evaluate_model.best_acc = 0.0
    evaluate_model.best_ap = 0.0

    # ===========================
    # TRAINING LOOP
    # ===========================
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        print(f"\n{'=' * 60}")
        print(f"--- Epoch {epoch + 1}/{opt.niter} ---")
        print(f"{'=' * 60}")

        model.train()
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        num_batches = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1

            try:
                model.set_input(data)
                model.optimize_parameters()
            except Exception as e:
                print(f"[ERROR] Failed at step {model.total_steps}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            loss_val = float(model.loss)
            acc_val = float(model.batch_acc)
            if torch.isnan(torch.tensor(loss_val)):
                print(
                    f"⚠️ NaN loss detected at step {model.total_steps}, skipping batch.")
                continue

            epoch_loss_sum += loss_val
            epoch_acc_sum += acc_val
            num_batches += 1

            # logging
            if model.total_steps % opt.loss_freq == 0:
                print(
                    f"[Step {model.total_steps:6d}] "
                    f"Loss: {loss_val:.6f} | Batch Acc: {acc_val:.4f} | LR: {model.optimizer.param_groups[0]['lr']:.2e}"
                )
                train_writer.add_scalar("loss", loss_val, model.total_steps)
                train_writer.add_scalar(
                    "batch_acc", acc_val, model.total_steps)
                train_writer.add_scalar(
                    "learning_rate", model.optimizer.param_groups[0]["lr"], model.total_steps
                )

            # intermediate save + val
            if model.total_steps % opt.save_latest_freq == 0:
                print(f"\n💾 Saving latest model (step {model.total_steps})")
                model.save_networks("latest")
                val_acc = evaluate_model(epoch)
                if val_acc is not None:
                    model.update_learning_rate(val_acc)
                print()

        # end of epoch summary
        avg_epoch_loss = epoch_loss_sum / max(num_batches, 1)
        avg_epoch_acc = epoch_acc_sum / max(num_batches, 1)
        epoch_losses.append(avg_epoch_loss)
        epoch_accs.append(avg_epoch_acc)

        print(f"\n{'─' * 60}")
        print(f"📊 Epoch {epoch + 1} Summary:")
        print(f"   Avg Train Loss: {avg_epoch_loss:.6f}")
        print(f"   Avg Train Batch Acc: {avg_epoch_acc:.4f}")
        print(f"{'─' * 60}\n")

        model.print_label_stats()

        # save checkpoints
        print(f"💾 Saving checkpoint at end of epoch {epoch}")
        model.save_networks("latest")
        model.save_networks(epoch)

        # validation
        acc = evaluate_model(epoch)
        if acc is not None:
            model.update_learning_rate(acc)

        # early stopping
        early_stopping(acc, model)
        if early_stopping.early_stop:
            print("\n🛑 Early stopping triggered — ending training.")
            print(
                f"   Best validation accuracy: {evaluate_model.best_acc:.6f}")
            print(f"   Best validation AP: {evaluate_model.best_ap:.6f}")
            break

        epoch_time = time.time() - epoch_start_time
        print(
            f"\n⏱️ Epoch {epoch + 1} finished in {epoch_time:.2f}s ({epoch_time / 60:.1f} min)")

    # ===========================
    # TRAINING COMPLETED
    # ===========================
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"   Final best accuracy: {evaluate_model.best_acc:.6f}")
    print(f"   Final best AP: {evaluate_model.best_ap:.6f}")
    print("=" * 60)

    train_writer.close()
    val_writer.close()
