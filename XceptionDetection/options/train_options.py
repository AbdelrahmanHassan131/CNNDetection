import argparse
import os
import torch


class TrainOptions:
    """Training options for Xception"""
    
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Data options
        parser.add_argument('--dataroot', default='./dataset/', help='path to dataset')
        parser.add_argument('--classes', default='0_real,1_fake', help='classes to train on (comma separated)')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=333, help='resize images to this size')
        parser.add_argument('--cropSize', type=int, default=299, help='crop images to this size (Xception expects 299)')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        
        # Augmentation options
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip images')
        parser.add_argument('--no_crop', action='store_true', help='if specified, do not crop images')
        parser.add_argument('--no_resize', action='store_true', help='if specified, do not resize images')
        
        # Model options
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='xception_experiment', help='experiment name')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models saved here')
        
        # Training options
        parser.add_argument('--optim', type=str, default='adam', help='optimizer [adam, sgd]')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term for adam')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for weight init')
        parser.add_argument('--niter', type=int, default=100, help='# of epochs')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load')
        parser.add_argument('--continue_train', action='store_true', help='continue training from checkpoint')
        
        # Logging options
        parser.add_argument('--loss_freq', type=int, default=100, help='frequency of showing loss')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving latest model')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints')
        parser.add_argument('--earlystop_epoch', type=int, default=5, help='early stopping patience')
        
        # Validation options
        parser.add_argument('--train_split', type=str, default='train', help='train split name')
        parser.add_argument('--val_split', type=str, default='val', help='validation split name')
        parser.add_argument('--class_bal', action='store_true', help='use balanced sampling')
        parser.add_argument('--serial_batches', action='store_true', help='take images in order')
        
        self.initialized = True
        return parser

    def parse(self, print_options=True):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()
        opt.isTrain = True
        
        # Parse GPU IDs
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        
        # Parse classes
        opt.classes = opt.classes.split(',')
        
        if print_options:
            self.print_options(opt)
        
        return opt

    def print_options(self, opt):
        message = '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            message += f'{k:>25}: {v:<30}\n'
        message += '----------------- End -------------------'
        print(message)
        
        # Save options to disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
