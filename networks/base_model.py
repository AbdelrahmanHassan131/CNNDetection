import os
import torch
import torch.nn as nn
from torch.nn import init


class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = True
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # Set device
        if torch.cuda.is_available() and getattr(opt, 'gpu_ids', None) and len(opt.gpu_ids) > 0:
            self.device = torch.device(f'cuda:{opt.gpu_ids[0]}')
            print(f"🎮 Using GPU {opt.gpu_ids[0]} as primary device")
        else:
            self.device = torch.device('cpu')
            print("⚠️ Using CPU (no GPU available)")

    def to_device(self, x, non_blocking=True):
        """Move tensor or dict/list of tensors to model device safely."""
        if torch.is_tensor(x):
            return x.to(self.device, non_blocking=non_blocking)
        if isinstance(x, dict):
            return {k: self.to_device(v, non_blocking=non_blocking) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self.to_device(v, non_blocking=non_blocking) for v in x)
        return x

    def save_networks(self, epoch):
        os.makedirs(self.save_dir, exist_ok=True)
        save_filename = f'model_epoch_{epoch}.pth'
        save_path = os.path.join(self.save_dir, save_filename)

        # Extract state dict (handle DataParallel)
        model_obj = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        state = {
            'model_state': model_obj.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'total_steps': self.total_steps,
            'opt': vars(self.opt) if hasattr(self, 'opt') else None
        }
        if hasattr(self, 'scheduler'):
            state['scheduler_state'] = self.scheduler.state_dict()

        torch.save(state, save_path)
        print(f"💾 Saved checkpoint: {save_path}")

    def load_networks(self, epoch):
        load_filename = f'model_epoch_{epoch}.pth'
        load_path = os.path.join(self.save_dir, load_filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        print(f'🔄 loading the model from {load_path}')
        state = torch.load(load_path, map_location=self.device)

        model_state = state.get('model_state', None)
        if model_state is None:
            raise KeyError("Checkpoint missing 'model_state' key")

        # strip 'module.' if necessary
        new_state = {}
        for k, v in model_state.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state[name] = v

        model_obj = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        model_obj.load_state_dict(new_state, strict=False)
        print("✅ Model weights loaded")

        # optimizer
        if self.isTrain and (not getattr(self.opt, 'new_optim', False)) and state.get('optimizer_state') is not None:
            try:
                self.optimizer.load_state_dict(state['optimizer_state'])
                # move optimizer tensors to correct device
                for group in self.optimizer.state.values():
                    for k, v in group.items():
                        if torch.is_tensor(v):
                            group[k] = v.to(self.device)
                print("✅ Optimizer state loaded")
            except Exception as e:
                print(f"⚠️ Could not load optimizer state: {e}")

        # scheduler
        if self.isTrain and hasattr(self, 'scheduler') and state.get('scheduler_state') is not None:
            try:
                self.scheduler.load_state_dict(state['scheduler_state'])
                print("✅ Scheduler state loaded")
            except Exception as e:
                print(f"⚠️ Could not load scheduler state: {e}")

        self.total_steps = state.get('total_steps', self.total_steps)

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
            try:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
            except Exception:
                pass
    net.apply(init_func)
