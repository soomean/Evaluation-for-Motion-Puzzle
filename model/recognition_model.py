import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import rec_networks


class RecognitionModel(nn.Module):
    def __init__(self, opt):
        super(RecognitionModel, self).__init__()
        self.opt = opt
        self.mode = opt.mode
        self.device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")
        self.save_dir = opt.save_dir

        if opt.criteria == 'content':
            self.recognizer = rec_networks.Recognizer(input_nc=3, num_class=opt.num_contents)
        elif opt.criteria == 'style':
            self.recognizer = rec_networks.Recognizer(input_nc=3, num_class=opt.num_styles)
        
        if self.mode == 'recog':
            self.optimizer = self.set_optimizer(optim_type='Adam')
        
        self.to(self.device)

    def set_optimizer(self, optim_type='SGD'):
        if optim_type == 'SGD':
            optimizer = torch.optim.SGD(
                self.recognizer.parameters(),
                lr=self.opt.lr,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay)
        elif optim_type == 'Adam':
            optimizer = torch.optim.Adam(
                self.recognizer.parameters(),
                lr=self.opt.lr,
                betas=(self.opt.beta1, self.opt.beta2),
                weight_decay=self.opt.weight_decay)
        else:
            raise ValueError()

        return optimizer

    def reset_grad(self):
        self.optimizer.zero_grad()

    def get_current_iter(self):
        return self.current_iter

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']

        return lr

    def print_networks(self):
        save_path = os.path.join(self.save_dir, 'net.txt')
        with open(save_path, 'w') as nets_f:
            rec_networks.print_network(self.recognizer, nets_f)

    def save_networks(self, iter=None, latest=False):
        if latest:
            save_filename = 'latest_checkpoint.pth'
        else:
            save_filename = '%d_checkpoint.pth' % iter
        save_path = os.path.join(self.save_dir, save_filename)
        print('Saving the model into %s...' % save_path)

        checkpoint = {'iter': iter}
        checkpoint['model_state_dict'] = self.recognizer.state_dict()
        checkpoint['optim_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, save_path)

    def load_networks(self, iter=None):
        if iter is not None:
            load_filename = '%d_checkpoint.pth' % iter
        else:
            load_filename = 'latest_checkpoint.pth'
        load_path = os.path.join(self.save_dir, load_filename)

        checkpoint = torch.load(load_path, map_location='cuda:0')
        self.recognizer.load_state_dict(checkpoint['model_state_dict'])

        if self.mode == 'recog':
            self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.opt.lr
        
        self.current_iter = checkpoint['iter']
