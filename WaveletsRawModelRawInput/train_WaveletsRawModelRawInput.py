"""
Training script for Wavelet Packet model - Original Wolter et al. 2022
Clean implementation without augmentation
"""

import os
import sys
import time
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter

from validate import validate
from earlystop import EarlyStopping
from WaveletsRawModelRawInput.trainer_WaveletsRawModelRawInput import WaveletPacketTrainer
from WaveletsRawModelRawInput.data_wavelets import create_dataloader_wavelet
from options.train_options import TrainOptions


def get_val_opt():
    """Get validation options"""
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    
    # Disable augmentation for validation
    val_opt.blur_prob = 0.0
    val_opt.jpg_prob = 0.0
    
    # Wavelet parameters
    val_opt.wavelet_type = 'haar'
    val_opt.wavelet_level = 3
    val_opt.wavelet_mode = 'reflect'
    val_opt.use_log_packets = True
    
    # Override for standard size
    val_opt.loadSize = 256
    val_opt.cropSize = 224
    
    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    
    # Override settings for Wavelet model
    opt.loadSize = 256
    opt.cropSize = 224
    
    # Wavelet parameters
    opt.wavelet_type = 'haar'
    opt.wavelet_level = 3
    opt.wavelet_mode = 'reflect'
    opt.use_log_packets = True
    
    # Disable augmentation for clean training
    opt.blur_prob = 0.0
    opt.jpg_prob = 0.0
    opt.no_flip = True
    
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()
    
    print("Creating wavelet dataloader...")
    data_loader = create_dataloader_wavelet(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % (dataset_size * opt.batch_size))
    
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    print("Initializing model...")
    model = WaveletPacketTrainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    
    print("Starting training...")
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size
            
            model.set_input(data)
            model.optimize_parameters()
            
            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)
            
            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')
        
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        
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
