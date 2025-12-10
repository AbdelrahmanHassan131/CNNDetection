import os
import torch


class BaseModel:
    """Base class for models"""
    
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.total_steps = 0

    def save_networks(self, epoch):
        save_path = os.path.join(self.save_dir, f'{epoch}_net.pth')
        torch.save(self.model.state_dict(), save_path)
        
        # Save optimizer state
        if hasattr(self, 'optimizer'):
            save_path_opt = os.path.join(self.save_dir, f'{epoch}_opt.pth')
            torch.save(self.optimizer.state_dict(), save_path_opt)

    def load_networks(self, epoch):
        load_path = os.path.join(self.save_dir, f'{epoch}_net.pth')
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f'Loaded model from {load_path}')
        else:
            print(f'Warning: No model found at {load_path}')

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
