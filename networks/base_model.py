import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = True
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # 🆕 IMPROVED: Better multi-GPU handling
        if torch.cuda.is_available() and len(opt.gpu_ids) > 0:
            self.device = torch.device(f'cuda:{opt.gpu_ids[0]}')
            print(f"🎮 Using GPU {opt.gpu_ids[0]} as primary device")
        else:
            self.device = torch.device('cpu')
            print("⚠️ Using CPU (no GPU available)")

        # 🆕 IMPROVED: Move model to device and setup DataParallel AFTER model is created
        # This is done in child class __init__ after model definition

    def setup_multi_gpu(self):
        """
        Call this method AFTER defining self.model in child class
        to enable multi-GPU training
        """
        if hasattr(self, 'model'):
            self.model.to(self.device)

            # 🆕 IMPROVED: Enable DataParallel if multiple GPUs
            if len(self.opt.gpu_ids) > 1 and torch.cuda.device_count() > 1:
                print(f"🚀 DataParallel enabled on GPUs: {self.opt.gpu_ids}")
                print(f"   Available GPUs: {torch.cuda.device_count()}")
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.opt.gpu_ids
                )
                print(
                    f"   Batch will be split across {len(self.opt.gpu_ids)} GPUs")
            else:
                print(f"ℹ️ Single GPU mode (only 1 GPU specified or available)")

    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)
        os.makedirs(self.save_dir, exist_ok=True)

        # 🆕 IMPROVED: Handle DataParallel models correctly
        state_dict = {
            'model': (self.model.module if isinstance(self.model, nn.DataParallel) else self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        # Save scheduler state if it exists
        if hasattr(self, 'scheduler'):
            state_dict['scheduler'] = self.scheduler.state_dict()

        torch.save(state_dict, save_path)

    def load_networks(self, epoch):
        load_filename = f'model_epoch_{epoch}.pth'
        load_path = os.path.join(self.save_dir, load_filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        print(f'loading the model from {load_path}')
        # 🆕 IMPROVED: Load to correct device
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        model_state = state_dict['model']

        # 🆕 IMPROVED: Handle models saved with or without DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state.items():
            # Remove 'module.' prefix if exists
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v

        # 🆕 IMPROVED: Load into correct model (with or without DataParallel)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(new_state_dict, strict=False)
        else:
            self.model.load_state_dict(new_state_dict, strict=False)

        self.total_steps = state_dict.get('total_steps', 0)

        # Only load optimizer state if we are continuing training
        if self.isTrain and not self.opt.new_optim and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # 🆕 IMPROVED: Move optimizer state to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            # Update learning rate from options
            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

        # Load scheduler state if it exists
        if self.isTrain and hasattr(self, 'scheduler') and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
