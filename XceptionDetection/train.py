import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


def get_val_opt():
    """Get validation options (modified from training options)"""
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = f'{val_opt.dataroot}/{val_opt.val_split}/'
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.dataroot = f'{opt.dataroot}/{opt.train_split}/'
    val_opt = get_val_opt()

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print(f'#training images = {dataset_size * opt.batch_size}')

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(f"Train loss: {model.loss:.4f} at step: {model.total_steps}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print(f'saving the latest model (epoch {epoch}, total_steps {model.total_steps})')
                model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print(f"(Val @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f}")

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()

    train_writer.close()
    val_writer.close()
    print("Training completed!")
