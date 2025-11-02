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
    # 🆕 IMPROVED: Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # 🆕 ADDED: Print GPU information
    print("="*60)
    print("🎮 GPU INFORMATION")
    print("="*60)
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ No CUDA GPUs available!")
    print("="*60)
    print()

    opt = TrainOptions().parse()
    opt.dataroot = f"{opt.dataroot}/{opt.train_split}/"
    val_opt = get_val_opt()

    # === Data loader ===
    print("📦 Creating data loader...")
    data_loader = create_dataloader(opt)
    print(f"✓ Training batches per epoch: {len(data_loader)}")
    print(f"✓ Total images per epoch: {len(data_loader) * opt.batch_size}")

    # 🆕 IMPROVED: Calculate effective batch size with multi-GPU
    if len(opt.gpu_ids) > 1:
        effective_batch = opt.batch_size * len(opt.gpu_ids)
        print(f"✓ Effective batch size (multi-GPU): {effective_batch}")
        print(f"  ({opt.batch_size} per GPU × {len(opt.gpu_ids)} GPUs)")
    print()

    # === Logging ===
    train_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(
        opt.checkpoints_dir, opt.name, "val"))

    # === Model ===
    model = Wavelet_ResNet_Trainer(opt)

    # Early stopping configuration
    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch,
        delta=0.001,
        verbose=True
    )

    print(f"🚀 Starting Wavelet-ResNet50 training for {opt.niter} epochs...")
    print(f"📋 Initial LR: {opt.lr:.2e}")
    print(f"📋 Batch size per GPU: {opt.batch_size}")
    if len(opt.gpu_ids) > 1:
        print(
            f"📋 Total effective batch size: {opt.batch_size * len(opt.gpu_ids)}")
    print(f"📋 Device: {model.device}")
    print(f"📋 Data augmentation: {'Enabled' if opt.data_aug else 'Disabled'}")
    print(f"📋 Class balancing: {'Enabled' if opt.class_bal else 'Disabled'}")
    print()

    # Track best metrics
    best_acc = 0.0
    best_ap = 0.0

    # === Enhanced validation function ===
    def evaluate_model(epoch):
        global best_acc, best_ap

        model.eval()
        try:
            acc_ap = validate_wavelet(model.model, val_opt)

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

        # Log to tensorboard
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)

        # Track improvements
        improved = ""
        if acc > best_acc:
            best_acc = acc
            improved += " 🆙 ACC"
        if ap > best_ap:
            best_ap = ap
            improved += " 🆙 AP"

        print(f"(Val @ epoch {epoch}) acc: {acc:.6f}; ap: {ap:.6f}{improved}")
        print(
            f"              Best so far → acc: {best_acc:.6f}; ap: {best_ap:.6f}")

        model.train()
        return acc

    # === Training loop ===
    epoch_losses = []
    epoch_accs = []

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"--- Epoch {epoch + 1}/{opt.niter} ---")
        print(f"{'='*60}")

        model.train()
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        num_batches = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            try:
                model.set_input(data)
            except Exception as e:
                print(f"[ERROR] Failed at step {model.total_steps}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            # === Train Step ===
            model.optimize_parameters()

            # Accumulate metrics
            epoch_loss_sum += float(model.loss)
            epoch_acc_sum += float(model.batch_acc)
            num_batches += 1

            # Logging
            if model.total_steps % opt.loss_freq == 0:
                try:
                    loss_val = float(model.loss)
                    batch_acc = float(model.batch_acc)
                except Exception:
                    loss_val = model.loss
                    batch_acc = 0.0

                print(f"[Step {model.total_steps:6d}] "
                      f"Loss: {loss_val:.6f} | "
                      f"Batch Acc: {batch_acc:.4f} | "
                      f"LR: {model.optimizer.param_groups[0]['lr']:.2e}")

                train_writer.add_scalar('loss', model.loss, model.total_steps)
                train_writer.add_scalar(
                    'batch_acc', model.batch_acc, model.total_steps)
                train_writer.add_scalar('learning_rate',
                                        model.optimizer.param_groups[0]['lr'],
                                        model.total_steps)

            # Periodic validation during epoch
            if model.total_steps % opt.save_latest_freq == 0:
                print(f"\n💾 Saving latest model (step {model.total_steps})")
                model.save_networks('latest')
                val_acc = evaluate_model(epoch)

                # Update learning rate based on validation
                if val_acc is not None:
                    model.update_learning_rate(val_acc)
                print()

        # === End of epoch summary ===
        avg_epoch_loss = epoch_loss_sum / num_batches
        avg_epoch_acc = epoch_acc_sum / num_batches
        epoch_losses.append(avg_epoch_loss)
        epoch_accs.append(avg_epoch_acc)

        print(f"\n{'─'*60}")
        print(f"📊 Epoch {epoch + 1} Summary:")
        print(f"   Avg Train Loss: {avg_epoch_loss:.6f}")
        print(f"   Avg Train Batch Acc: {avg_epoch_acc:.4f}")
        print(f"{'─'*60}\n")

        # Print label distribution
        model.print_label_stats()

        # === Save epoch checkpoint & validation ===
        print(f"💾 Saving checkpoint at end of epoch {epoch}")
        model.save_networks('latest')
        model.save_networks(epoch)

        # Final validation for the epoch
        acc = evaluate_model(epoch)

        # Update learning rate
        if acc is not None:
            model.update_learning_rate(acc)

        # Early stopping check
        early_stopping(acc, model)
        if early_stopping.early_stop:
            print("\n🛑 Early stopping triggered — ending training.")
            print(f"   Best validation accuracy: {best_acc:.6f}")
            print(f"   Best validation AP: {best_ap:.6f}")
            break

        epoch_time = time.time() - epoch_start_time
        print(f"\n⏱️ Epoch {epoch + 1} finished in {epoch_time:.2f}s "
              f"({epoch_time/60:.1f} min)")

    # === Training complete ===
    print("\n" + "="*60)
    print("✅ Training completed successfully!")
    print(f"   Final best accuracy: {best_acc:.6f}")
    print(f"   Final best AP: {best_ap:.6f}")
    print("="*60)

    train_writer.close()
    val_writer.close()
